"""LoRA mixture-of-experts layers for regime-aware adaptation (RegMoE).

The base model stays frozen and every layer applies a low-rank additive
correction, so the pretrained output is always preserved:

    output = base(x) + (alpha / r) * sum_i w_i * B_i(A_i(x))

Two injectable layer types:

  - ``LoRAMoELayer``: N experts with a learned soft router over tokens, plus a
    load-balancing loss against expert collapse.
  - ``TaskConditionedLoRAMoELayer``: the RegMoE layer. Same mixture, plus a
    learned per-regime gate bias, ``logits = gate(x) + b(regime)``. The bias is
    initialised to zero, so a model evaluated without regime ids behaves exactly
    like the plain soft router above.

Use ``replace_linear_with_moe`` to inject either type into a frozen backbone.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _aggregate_for_gate(x, b_merge, l_merge, aggregation="mean"):
    """Produce [B, D] sample representation from [B, L, D] or [B*L, D] input."""
    if x.dim() == 3:
        if aggregation == "mean":
            return x.mean(dim=1)
        elif aggregation == "max":
            return x.max(dim=1).values
        elif aggregation == "last":
            return x[:, -1, :]
    elif x.dim() == 2 and b_merge is not None and l_merge is not None:
        x_3d = x.view(b_merge, l_merge, -1)
        return _aggregate_for_gate(x_3d, None, None, aggregation)
    return x.mean(dim=0, keepdim=True)


class LoRAExpert(nn.Module):
    """Single LoRA expert: W_delta = B @ A."""

    def __init__(self, in_features: int, out_features: int, rank: int = 8, dtype=torch.float32):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False, dtype=dtype)
        self.B = nn.Linear(rank, out_features, bias=False, dtype=dtype)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x))


# ---------------------------------------------------------------------------
# LoRA mixture of experts with a soft content router
# ---------------------------------------------------------------------------

class LoRAMoELayer(nn.Module):
    """
    LoRAMoE: multiple LoRA experts with a learned soft router.
    output = base(x) + (alpha/r) * sum_i(w_i * B_i @ A_i @ x)
    where w = softmax(x @ W_gate)
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        num_experts: int = 4,
        rank: int = 8,
        alpha: float = 16.0,
        expert_dropout: float = 0.0,
        gate_temperature: float = 1.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.num_experts = num_experts
        self.rank = rank
        self.scaling = alpha / rank
        self.expert_dropout = expert_dropout
        self.gate_temperature = gate_temperature

        in_f = base_layer.in_features
        out_f = base_layer.out_features
        dtype = base_layer.weight.dtype

        self.experts = nn.ModuleList([
            LoRAExpert(in_f, out_f, rank, dtype=dtype) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(in_f, num_experts, bias=False, dtype=dtype)

        for p in base_layer.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        gate_weights = F.softmax(self.gate(x) / self.gate_temperature, dim=-1)

        if self.training and self.expert_dropout > 0 and self.num_experts > 1:
            drop_mask = torch.ones(self.num_experts, device=x.device, dtype=gate_weights.dtype)
            if torch.rand(1).item() < self.expert_dropout:
                drop_idx = torch.randint(0, self.num_experts, (1,)).item()
                drop_mask[drop_idx] = 0
            gate_weights = gate_weights * drop_mask
            gate_weights = gate_weights / (gate_weights.sum(dim=-1, keepdim=True) + 1e-8)

        if self.training:
            avg_weights = gate_weights.mean(dim=list(range(gate_weights.dim() - 1)))
            uniform = torch.ones_like(avg_weights) / self.num_experts
            self.last_load_balancing_loss = F.mse_loss(avg_weights, uniform)

        moe_out = torch.zeros_like(base_out)
        for i, expert in enumerate(self.experts):
            moe_out = moe_out + expert(x) * gate_weights[..., i].unsqueeze(-1)

        return base_out + self.scaling * moe_out


# ---------------------------------------------------------------------------
# Task-conditioned LoRA mixture of experts (RegMoE)
# ---------------------------------------------------------------------------

class TaskConditionedLoRAMoELayer(nn.Module):
    """
    Soft LoRA MoE with a regime-dependent gate bias: logits = gate(x) + b(regime).
    Regimes are the training labels in {0, 1, 2} (cruiser / maneuverer / reactor).

    The trainer sets ``_task_bc`` ([B]), ``_task_flat`` ([B*L]) and
    ``_b_merge`` / ``_l_merge`` before each forward. At inference no regime id
    is set, so the bias is unused and the layer reduces to the plain soft
    router (the bias is zero-initialised).
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        num_experts: int = 4,
        rank: int = 8,
        alpha: float = 16.0,
        num_task_types: int = 3,
        routing_level: str = "token",
        aggregation: str = "mean",
    ):
        super().__init__()
        self.base_layer = base_layer
        self.num_experts = num_experts
        self.rank = rank
        self.scaling = alpha / rank
        self.num_task_types = num_task_types
        self.routing_level = routing_level
        self.aggregation = aggregation

        in_f = base_layer.in_features
        out_f = base_layer.out_features
        dtype = base_layer.weight.dtype

        self.experts = nn.ModuleList([
            LoRAExpert(in_f, out_f, rank, dtype=dtype) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(in_f, num_experts, bias=False, dtype=dtype)
        self.task_gate_bias = nn.Embedding(num_task_types, num_experts, dtype=dtype)
        nn.init.zeros_(self.task_gate_bias.weight)

        for p in base_layer.parameters():
            p.requires_grad = False

        self._task_bc = None
        self._task_flat = None
        self._b_merge = None
        self._l_merge = None
        self._cached_gate_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)

        if self.routing_level == "sample":
            return self._forward_sample(x, base_out)
        return self._forward_token(x, base_out)

    def _forward_sample(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        """Sample-level routing: one gate decision per sample, broadcast to all tokens."""
        # 1. Aggregate x → [B, D]
        x_agg = _aggregate_for_gate(x, self._b_merge, self._l_merge, self.aggregation)

        # 2. Gate on sample representation → [B, N]
        gate_logits = self.gate(x_agg)
        if self._task_bc is not None and self._task_bc.device == x.device:
            gate_logits = gate_logits + self.task_gate_bias(self._task_bc)
        gate_weights = F.softmax(gate_logits, dim=-1)  # [B, N]

        # 3. Inference: cache gate weights from prefill, reuse in decode steps
        if not self.training:
            if x.dim() == 3 and x.shape[1] > 1:
                self._cached_gate_weights = gate_weights
            elif self._cached_gate_weights is not None:
                gate_weights = self._cached_gate_weights

        # 4. Training stats: already [B, N], no token-dim averaging needed
        if self.training:
            avg_w = gate_weights.mean(dim=0)  # [N]
            self.last_mean_gate_per_sample = gate_weights  # [B, N] directly
            uniform = torch.ones_like(avg_w) / self.num_experts
            self.last_load_balancing_loss = F.mse_loss(avg_w, uniform)

        # 5. Broadcast gate weights to token dimension
        if x.dim() == 3:
            gw = gate_weights.unsqueeze(1).expand(-1, x.shape[1], -1)  # [B, L, N]
        elif x.dim() == 2 and self._b_merge is not None:
            B, L = self._b_merge, self._l_merge
            gw = gate_weights.unsqueeze(1).expand(-1, L, -1).reshape(B * L, -1)  # [B*L, N]
        else:
            gw = gate_weights.unsqueeze(0).expand(x.shape[0], -1)

        moe_out = torch.zeros_like(base_out)
        for i, expert in enumerate(self.experts):
            moe_out = moe_out + expert(x) * gw[..., i].unsqueeze(-1)
        return base_out + self.scaling * moe_out

    def _forward_token(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        """Token-level routing with the optional regime bias."""
        gate_logits = self.gate(x)

        if self._task_bc is not None and self._task_bc.device == x.device:
            if x.dim() == 3:
                bias = self.task_gate_bias(self._task_bc)
                gate_logits = gate_logits + bias.unsqueeze(1)
            elif x.dim() == 2 and self._task_flat is not None:
                if self._task_flat.shape[0] == gate_logits.shape[0]:
                    gate_logits = gate_logits + self.task_gate_bias(self._task_flat)

        gate_weights = F.softmax(gate_logits, dim=-1)

        if self.training:
            if x.dim() == 3:
                avg_w = gate_weights.mean(dim=(0, 1))
                self.last_mean_gate_per_sample = gate_weights.mean(dim=1)
            elif (
                x.dim() == 2
                and self._b_merge is not None
                and self._l_merge is not None
                and x.shape[0] == self._b_merge * self._l_merge
            ):
                B, Lm = self._b_merge, self._l_merge
                gw = gate_weights.view(B, Lm, -1)
                avg_w = gw.mean(dim=(0, 1))
                self.last_mean_gate_per_sample = gw.mean(dim=1)
            else:
                dims = tuple(range(gate_weights.dim() - 1))
                avg_w = gate_weights.mean(dim=dims) if dims else gate_weights.mean(dim=0)
                self.last_mean_gate_per_sample = None

            uniform = torch.ones_like(avg_w) / self.num_experts
            self.last_load_balancing_loss = F.mse_loss(avg_w, uniform)

        moe_out = torch.zeros_like(base_out)
        for i, expert in enumerate(self.experts):
            moe_out = moe_out + expert(x) * gate_weights[..., i].unsqueeze(-1)
        return base_out + self.scaling * moe_out


# ---------------------------------------------------------------------------
# Utility: replace target layers in a model
# ---------------------------------------------------------------------------

def replace_linear_with_moe(
    model: nn.Module,
    method: str,
    target_modules: list,
    layers_to_transform: Optional[list] = None,
    **kwargs,
) -> nn.Module:
    """
    Replace specified Linear layers in the model with MoE-LoRA layers.

    Args:
        model: The base model (e.g., Qwen3-VL).
        method: "tcloramoe" for RegMoE, or "loramoe" for the plain soft router.
        target_modules: List of module name suffixes to replace (e.g., ["q_proj", "v_proj"]).
        layers_to_transform: List of layer indices to modify (None = all layers).
        **kwargs: Passed to the MoE layer constructor (rank, num_experts, etc.).

    Returns:
        The model with replaced layers.
    """
    layer_cls = {
        "loramoe": LoRAMoELayer,
        "tcloramoe": TaskConditionedLoRAMoELayer,
    }[method]

    replaced = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        should_replace = any(name.endswith(t) for t in target_modules)
        if not should_replace:
            continue

        if layers_to_transform is not None:
            layer_idx = _extract_layer_index(name)
            if layer_idx is None or layer_idx not in layers_to_transform:
                continue

        parent_name, attr_name = name.rsplit(".", 1)
        parent = dict(model.named_modules())[parent_name]

        new_layer = layer_cls(module, **kwargs)
        setattr(parent, attr_name, new_layer)
        replaced += 1

    print(f"[MoE] Replaced {replaced} layers with {method} (target={target_modules})")
    if replaced == 0:
        raise ValueError(
            "No target layers were replaced. Check target_modules and "
            "layers_to_transform against model.named_modules()."
        )
    return model


def _extract_layer_index(name: str) -> Optional[int]:
    """Extract transformer layer index from module name like 'model.layers.5.mlp.gate_proj'."""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None
