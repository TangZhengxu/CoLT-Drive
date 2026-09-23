"""SLERP interpolation between a pretrained VLM and a driving expert.

Rather than fine-tuning all the way toward the driving expert, take a small
spherical step from the pretrained weights and freeze the result. This keeps
most of the pretrained representation intact while moving toward driving
specialisation. KPA uses alpha = 0.08.

The merged checkpoint is what ``kpa.train`` should be pointed at.

Usage:
    python -m kpa.slerp_merge \
        --base Qwen/Qwen3-VL-2B-Instruct \
        --finetuned /path/to/driving_lora_adapter \
        --output /path/to/merged \
        --alphas 0.08

Each alpha writes to ``<output>_alpha<NNN>``, e.g. ``merged_alpha008``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os

import torch

from .backbone import load_vlm


def load_state_dict_from_adapter(base_name: str, adapter_path: str) -> dict[str, torch.Tensor]:
    """Merge a LoRA adapter into the base model and return the full state dict."""
    from peft import PeftModel

    base = load_vlm(base_name)
    model = PeftModel.from_pretrained(base, adapter_path)
    merged = model.merge_and_unload()
    state = {k: v.cpu().clone() for k, v in merged.state_dict().items()}
    del merged, model, base
    gc.collect()
    torch.cuda.empty_cache()
    return state


def load_state_dict_from_full(model_path: str) -> dict[str, torch.Tensor]:
    """Load a full (non-adapter) checkpoint and return its state dict."""
    model = load_vlm(model_path)
    state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return state


def _lerp(sd_a: dict, sd_b: dict, alpha: float) -> dict[str, torch.Tensor]:
    """Linear interpolation: (1 - alpha) * A + alpha * B."""
    return {key: (1.0 - alpha) * sd_a[key] + alpha * sd_b[key] for key in sd_a}


def _slerp_tensor(v0: torch.Tensor, v1: torch.Tensor, t: float, eps: float = 1e-8) -> torch.Tensor:
    """Spherical interpolation between two tensors, computed in float32.

    Falls back to linear interpolation when the vectors are degenerate or very
    nearly collinear, where the spherical formula is numerically unstable.
    """
    v0_flat = v0.float().flatten()
    v1_flat = v1.float().flatten()
    n0 = v0_flat.norm()
    n1 = v1_flat.norm()
    if n0 < eps or n1 < eps:
        return ((1.0 - t) * v0 + t * v1).to(v0.dtype)

    dot = torch.clamp(((v0_flat / n0) * (v1_flat / n1)).sum(), -1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    if so.abs() < eps:
        return ((1.0 - t) * v0 + t * v1).to(v0.dtype)

    result = (
        (torch.sin((1.0 - t) * omega) / so) * v0_flat
        + (torch.sin(t * omega) / so) * v1_flat
    )
    return result.reshape(v0.shape).to(v0.dtype)


def _slerp(sd_a: dict, sd_b: dict, alpha: float) -> dict[str, torch.Tensor]:
    """SLERP per parameter tensor; non-float tensors are selected, not blended."""
    out = {}
    for key in sd_a:
        if sd_a[key].dtype in (torch.float16, torch.bfloat16, torch.float32):
            out[key] = _slerp_tensor(sd_a[key], sd_b[key], alpha)
        else:
            out[key] = sd_b[key] if alpha > 0.5 else sd_a[key]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpolate a VLM toward a driving expert")
    parser.add_argument("--base", required=True, help="Pretrained model name or path")
    parser.add_argument("--finetuned", required=True, help="Driving expert: LoRA adapter or full model")
    parser.add_argument("--output", required=True, help="Output directory prefix")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.08],
                        help="Interpolation coefficients; KPA uses 0.08")
    parser.add_argument("--method", choices=["slerp", "linear"], default="slerp")
    parser.add_argument("--no-adapter", action="store_true",
                        help="Treat --finetuned as a full model rather than an adapter")
    args = parser.parse_args()
    if any(alpha < 0.0 or alpha > 1.0 for alpha in args.alphas):
        raise ValueError("Every alpha must be in the closed interval [0, 1]")

    print(f"Base:       {args.base}")
    print(f"Fine-tuned: {args.finetuned}")
    print(f"Method:     {args.method}   Alphas: {args.alphas}")

    print("\n=== Loading base model ===")
    sd_base = load_state_dict_from_full(args.base)

    print("\n=== Loading driving expert ===")
    if args.no_adapter:
        sd_expert = load_state_dict_from_full(args.finetuned)
    else:
        sd_expert = load_state_dict_from_adapter(args.base, args.finetuned)

    missing = set(sd_base) - set(sd_expert)
    extra = set(sd_expert) - set(sd_base)
    if missing or extra:
        raise ValueError(
            "Base and expert state dictionaries are incompatible: "
            f"{len(missing)} missing and {len(extra)} extra keys"
        )

    interpolate = _slerp if args.method == "slerp" else _lerp

    for alpha in args.alphas:
        print(f"\n=== Merging at alpha={alpha:.2f} ===")
        merged = interpolate(sd_base, sd_expert, alpha)

        out_dir = f"{args.output}_alpha{f'{alpha:.2f}'.replace('.', '')}"
        os.makedirs(out_dir, exist_ok=True)

        save_model = load_vlm(args.base)
        save_model.load_state_dict(merged, strict=True)
        save_model.save_pretrained(out_dir)

        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(args.base).save_pretrained(out_dir)

        with open(os.path.join(out_dir, "merge_config.json"), "w") as f:
            json.dump(
                {
                    "base": args.base,
                    "finetuned": args.finetuned,
                    "method": args.method,
                    "alpha": alpha,
                },
                f,
                indent=2,
            )

        print(f"  Saved to {out_dir}")
        del merged, save_model
        gc.collect()

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
