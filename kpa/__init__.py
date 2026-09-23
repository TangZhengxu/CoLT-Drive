"""KPA: regime-aware LoRA mixture-of-experts adaptation for driving VLMs.

Three ingredients, in the order they are applied:

1. A structured decision interface at the prompt level, so the model emits a
   parsable longitudinal/lateral action pair. See ``colt_drive.prompts``.
2. SLERP initialisation: interpolate the pretrained VLM toward a driving
   expert by a small step, then freeze the merged backbone. See
   ``kpa.slerp_merge``.
3. Regime-aware capacity: inject a LoRA mixture of experts whose gate is
   biased per behaviour regime. See ``kpa.moe_layers``.
"""

from .moe_layers import (
    LoRAExpert,
    LoRAMoELayer,
    TaskConditionedLoRAMoELayer,
    replace_linear_with_moe,
)

__all__ = [
    "LoRAExpert",
    "LoRAMoELayer",
    "TaskConditionedLoRAMoELayer",
    "replace_linear_with_moe",
]
