"""Backbone loading shared by training, merging, and inference."""

from __future__ import annotations

import json
import os

import torch


def _model_class(name: str):
    """Pick the right model class from the checkpoint's model_type."""
    model_type = ""
    config_path = os.path.join(os.path.expanduser(name), "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            model_type = json.load(f).get("model_type") or ""

    if "Qwen3-VL" in name or model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration as cls
    elif "Qwen2.5-VL" in name or model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration as cls
    else:
        from transformers import AutoModelForImageTextToText as cls
    return cls


def load_vlm(
    name: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
    device_map: str | None = None,
):
    """Load a vision-language model from a local path or hub id."""
    kwargs = {"torch_dtype": torch_dtype, "attn_implementation": attn_implementation}
    if device_map is not None:
        kwargs["device_map"] = device_map
    return _model_class(name).from_pretrained(name, **kwargs)
