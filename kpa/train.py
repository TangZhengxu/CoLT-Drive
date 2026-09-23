"""Training entry point for regime-aware LoRA mixture-of-experts (RegMoE).

The backbone stays frozen; only the injected MoE-LoRA parameters are trained.
The loss has three parts:

    total = answer_weight * LM_loss
          + lb_weight     * load_balancing_loss
          - sep_weight    * between_regime_gate_variance

The separation term is subtracted, i.e. it is *maximised*: experts are pushed to
route differently across behaviour regimes. Setting its weight to zero disables
that auxiliary objective; use ``moe.method: loramoe`` to remove regime bias.

Usage:
    torchrun --nproc_per_node=8 -m kpa.train --config kpa/configs/kpa_qwen3vl_2b.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
import torch.nn.functional as F
import yaml
from transformers import (
    AutoProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from .backbone import load_vlm
from .collator import DrivingCollator
from .data import DrivingDecisionDataset
from .loading import layer_kwargs_from_moe_cfg, moe_state_dict
from .moe_layers import (
    LoRAMoELayer,
    TaskConditionedLoRAMoELayer,
    replace_linear_with_moe,
)


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def load_backbone(cfg: dict):
    """Load the backbone. For KPA this is the SLERP-merged checkpoint produced
    by ``kpa.slerp_merge``, not the raw pretrained VLM."""
    name = cfg["model"]["name"]
    model = load_vlm(
        name,
        torch_dtype=getattr(torch, cfg["model"].get("torch_dtype", "bfloat16")),
        attn_implementation=cfg["model"].get("attn_implementation", "sdpa"),
    )
    return model, AutoProcessor.from_pretrained(name)


def freeze_backbone(model):
    for p in model.parameters():
        p.requires_grad = False
    print("Froze all backbone parameters")


def apply_regmoe(model, cfg: dict):
    """Replace target linears with MoE-LoRA layers."""
    moe_cfg = cfg["moe"]
    method = moe_cfg["method"]

    layer_kwargs = layer_kwargs_from_moe_cfg(moe_cfg)

    model = replace_linear_with_moe(
        model,
        method,
        moe_cfg["target_modules"],
        layers_to_transform=moe_cfg.get("layers_to_transform"),
        **layer_kwargs,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model


# ---------------------------------------------------------------------------
# Regime routing hooks
# ---------------------------------------------------------------------------

def clear_regime_hooks(model) -> None:
    """Drop per-step routing state.

    Called before each forward, never after: gradient checkpointing recomputes
    the forward during backward and needs the same hooks that the original
    forward saw.
    """
    for m in _unwrap(model).modules():
        if isinstance(m, TaskConditionedLoRAMoELayer):
            m._task_bc = None
            m._task_flat = None
            m._b_merge = None
            m._l_merge = None
            m._cached_gate_weights = None


def set_regime_hooks(core, regimes: torch.Tensor, batch: int, length: int) -> None:
    flat = regimes.unsqueeze(1).expand(-1, length).reshape(-1)
    for m in core.modules():
        if isinstance(m, TaskConditionedLoRAMoELayer):
            m._task_bc = regimes
            m._task_flat = flat
            m._b_merge = batch
            m._l_merge = length


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class RegMoETrainer(Trainer):
    """Adds the MoE auxiliary losses and answer-token weighting."""

    def __init__(
        self,
        *args,
        moe_method: str = "tcloramoe",
        answer_loss_weight: float = 1.0,
        lb_loss_weight: float = 0.1,
        separation_loss_weight: float = 0.05,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.moe_method = moe_method
        self.answer_loss_weight = float(answer_loss_weight)
        self.lb_loss_weight = float(lb_loss_weight)
        self.separation_loss_weight = float(separation_loss_weight)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        clear_regime_hooks(model)

        regimes = inputs.pop("regime", None)
        core = _unwrap(model)
        batch, length = inputs["input_ids"].shape
        if regimes is not None:
            set_regime_hooks(core, regimes, batch, length)

        labels = inputs.get("labels")

        if self.answer_loss_weight != 1.0 and labels is not None:
            forward_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            outputs = model(**forward_inputs, labels=None)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.shape)

            # The collator already masked everything except the target tokens,
            # so weighting the valid positions is a uniform rescale of the LM
            # loss. It behaves like a larger effective learning rate on the
            # answer, which is what answer_loss_weight=5 is doing.
            valid = shift_labels != -100
            if not valid.any():
                raise ValueError(
                    "Batch contains no target tokens. Check the target strings "
                    "and collator label masking."
                )
            loss = self.answer_loss_weight * per_token[valid].mean()
        else:
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        loss = loss + self._auxiliary_loss(core, regimes)
        return (loss, outputs) if return_outputs else loss

    def _auxiliary_loss(self, core, regimes):
        """Load balancing, plus between-regime gate separation for RegMoE."""
        if self.moe_method == "loramoe":
            losses = [
                m.last_load_balancing_loss
                for m in core.modules()
                if isinstance(m, LoRAMoELayer)
                and getattr(m, "last_load_balancing_loss", None) is not None
            ]
            if torch.is_grad_enabled() and any(not item.requires_grad for item in losses):
                raise RuntimeError(
                    "MoE auxiliary loss is detached. Use non-reentrant gradient "
                    "checkpointing or disable gradient checkpointing."
                )
            lb = sum(losses)
            return self.lb_loss_weight * lb

        lb = 0.0
        separation = 0.0
        counted_layers = 0
        for m in core.modules():
            if not isinstance(m, TaskConditionedLoRAMoELayer):
                continue
            if getattr(m, "last_load_balancing_loss", None) is not None:
                if torch.is_grad_enabled() and not m.last_load_balancing_loss.requires_grad:
                    raise RuntimeError(
                        "MoE auxiliary loss is detached. Use non-reentrant gradient "
                        "checkpointing or disable gradient checkpointing."
                    )
                lb += m.last_load_balancing_loss

            mean_gate = getattr(m, "last_mean_gate_per_sample", None)
            if regimes is None or mean_gate is None:
                continue
            per_regime = [
                mean_gate[regimes == r].mean(0)
                for r in range(m.num_task_types)
                if (regimes == r).any()
            ]
            if len(per_regime) >= 2:
                separation += torch.stack(per_regime).var(dim=0).mean()
                counted_layers += 1

        aux = self.lb_loss_weight * lb
        if counted_layers > 0 and self.separation_loss_weight > 0:
            # Subtracted, so training maximises how differently the gate routes
            # across regimes.
            aux = aux - self.separation_loss_weight * (separation / counted_layers)
        return aux


class SaveMoECallback(TrainerCallback):
    """Checkpoint only the MoE parameters, not the frozen backbone."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def _is_main(self) -> bool:
        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0

    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None or not self._is_main():
            return
        path = os.path.join(self.output_dir, f"moe-checkpoint-{state.global_step}")
        os.makedirs(path, exist_ok=True)
        torch.save(moe_state_dict(_unwrap(model)), os.path.join(path, "moe_weights.pt"))

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None or not self._is_main():
            return
        path = os.path.join(self.output_dir, f"checkpoint-epoch-{int(round(state.epoch))}")
        os.makedirs(path, exist_ok=True)
        torch.save(moe_state_dict(_unwrap(model)), os.path.join(path, "moe_weights.pt"))
        print(f"Saved epoch {int(round(state.epoch))} MoE weights to {path}")


def build_training_args(cfg: dict) -> TrainingArguments:
    t = cfg["training"]
    return TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        num_train_epochs=t["num_train_epochs"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t.get("warmup_ratio", 0.05),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        weight_decay=t.get("weight_decay", 0.01),
        bf16=t.get("bf16", True),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=t.get("dataloader_num_workers", 4),
        logging_steps=t.get("logging_steps", 50),
        save_strategy="steps",
        save_steps=t.get("save_steps", 100),
        save_total_limit=1,
        remove_unused_columns=False,
        report_to="none",
        max_steps=t.get("max_steps", -1),
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=True,
        seed=t.get("seed", 42),
    )


def save_adapter(model, cfg: dict, config_path: str) -> None:
    """Write the adapter in the layout expected by ``kpa.loading``."""
    final_dir = os.path.join(cfg["output_dir"], "final_moe")
    os.makedirs(final_dir, exist_ok=True)

    state = moe_state_dict(_unwrap(model))
    torch.save(state, os.path.join(final_dir, "moe_weights.pt"))

    with open(os.path.join(final_dir, "moe_config.json"), "w") as f:
        json.dump(dict(cfg.get("moe", {})), f, indent=2)
    shutil.copy2(config_path, os.path.join(final_dir, "training_config.yaml"))

    print(f"Saved adapter ({len(state)} tensors) + config to {final_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a RegMoE adapter")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model, processor = load_backbone(cfg)
    freeze_backbone(model)
    model = apply_regmoe(model, cfg)
    if cfg["training"].get("gradient_checkpointing", True):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    data_cfg = cfg["data"]
    dataset = DrivingDecisionDataset(
        manifest=data_cfg["manifest"],
        root=data_cfg.get("root"),
        max_samples=data_cfg.get("max_samples"),
    )
    print(f"Dataset: {len(dataset)} samples")

    trainer = RegMoETrainer(
        model=model,
        args=build_training_args(cfg),
        train_dataset=dataset,
        data_collator=DrivingCollator(processor),
        callbacks=[SaveMoECallback(cfg["output_dir"])],
        moe_method=cfg["moe"]["method"],
        answer_loss_weight=cfg.get("answer_loss_weight", 1.0),
        lb_loss_weight=cfg["moe"].get("lb_loss_weight", 0.1),
        separation_loss_weight=cfg["moe"].get("separation_loss_weight", 0.05),
    )

    trainer.train(resume_from_checkpoint=True if args.resume else None)

    if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0:
        save_adapter(model, cfg, args.config)


if __name__ == "__main__":
    main()
