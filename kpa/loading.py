"""Shared configuration and state selection for saving and loading RegMoE adapters.

For inference, rebuilds the MoE architecture from the adapter's
``moe_config.json``, then partially loads ``moe_weights.pt`` on the base model.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Set

import torch

from .regimes import NUM_REGIMES

# Shared by adapter saving and loading.
# NOTE: ".gate." (with dots) avoids matching base model gate_proj weights.
MOE_STATE_TAGS = (
    "expert",
    ".gate.",
    "task_gate_bias",
)


def moe_param_keys(state_dict_keys: Set[str]) -> Set[str]:
    return {k for k in state_dict_keys if any(tag in k for tag in MOE_STATE_TAGS)}


def moe_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Select adapter tensors from an unwrapped model, preserving state-dict order."""
    return {
        k: v for k, v in model.state_dict().items()
        if any(tag in k for tag in MOE_STATE_TAGS)
    }


def layer_kwargs_from_moe_cfg(moe_cfg: Dict[str, Any]) -> Dict[str, Any]:
    method = moe_cfg["method"]
    if method == "loramoe":
        return {
            "num_experts": moe_cfg.get("num_experts", 4),
            "rank": moe_cfg.get("rank", 8),
            "alpha": moe_cfg.get("alpha", 16.0),
            "expert_dropout": moe_cfg.get("expert_dropout", 0.0),
            "gate_temperature": moe_cfg.get("gate_temperature", 1.0),
        }
    if method == "tcloramoe":
        return {
            "num_experts": moe_cfg.get("num_experts", 4),
            "rank": moe_cfg.get("rank", 8),
            "alpha": moe_cfg.get("alpha", 16.0),
            "num_task_types": moe_cfg.get("num_task_types", NUM_REGIMES),
            "routing_level": moe_cfg.get("routing_level", "token"),
            "aggregation": moe_cfg.get("aggregation", "mean"),
        }
    raise ValueError(f"Unknown MoE method: {method}")


def attach_moe_weights(
    base_model: torch.nn.Module,
    moe_cfg: Dict[str, Any],
    moe_weights_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True,
) -> torch.nn.Module:
    """
    Replace linears with MoE layers, cast to dtype, load partial moe_weights.pt, move to device.
    """
    from .moe_layers import replace_linear_with_moe

    method = moe_cfg["method"]
    target_modules = moe_cfg["target_modules"]
    layers_to_transform = moe_cfg.get("layers_to_transform", None)
    layer_kwargs = layer_kwargs_from_moe_cfg(moe_cfg)

    model = replace_linear_with_moe(
        base_model,
        method,
        target_modules,
        layers_to_transform=layers_to_transform,
        **layer_kwargs,
    )
    model = model.to(dtype=dtype)

    state_dict = torch.load(
        moe_weights_path, map_location="cpu", weights_only=True
    )
    saved_keys = set(state_dict.keys())
    model_keys = set(model.state_dict().keys())
    moe_keys_model = moe_param_keys(model_keys)

    missing_in_ckpt = moe_keys_model - saved_keys
    unexpected_in_model = saved_keys - model_keys

    if missing_in_ckpt or unexpected_in_model:
        raise ValueError(
            "Adapter architecture does not match its checkpoint: "
            f"{len(missing_in_ckpt)} missing MoE keys, "
            f"{len(unexpected_in_model)} unknown checkpoint keys. "
            "Check the backbone and moe_config.json."
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    moe_still_missing = [k for k in missing if any(t in k for t in MOE_STATE_TAGS)]
    if moe_still_missing:
        raise ValueError(
            f"Adapter load left {len(moe_still_missing)} MoE keys unresolved"
        )
    if verbose:
        # Global missing/unexpected counts (checkpoint is partial — expect large missing)
        print(
            f"[MoE] load_state_dict: strict=False | total missing keys: {len(missing)} | "
            f"unexpected: {len(unexpected)}"
        )
        if unexpected:
            print(f"[MoE] unexpected (sample): {sorted(unexpected)[:8]}")

    return model.to(device)


def try_load_moe_from_adapter_dir(
    base_model: torch.nn.Module,
    adapter_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True,
) -> Optional[torch.nn.Module]:
    """
    If adapter_path contains moe_weights.pt + moe_config.json, attach MoE and return model.
    Otherwise return None.
    """
    moe_weights_path = os.path.join(adapter_path, "moe_weights.pt")
    moe_config_path = os.path.join(adapter_path, "moe_config.json")
    if not (os.path.exists(moe_weights_path) and os.path.exists(moe_config_path)):
        return None
    with open(moe_config_path) as f:
        moe_cfg = json.load(f)
    model = attach_moe_weights(
        base_model, moe_cfg, moe_weights_path, device, dtype=dtype, verbose=verbose
    )
    if verbose:
        print(f"[MoE] Loaded custom MoE weights from {adapter_path}")
    return model
