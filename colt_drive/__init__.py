"""CoLT-Drive: a counterfactual long-tail benchmark for driving decisions.

The benchmark holds the driving context fixed (road geometry, ego pose,
navigation command, motion history) and varies only the affordance in the
scene, then asks a single question: does the decision update when it should,
and does it stay put when it should not.

Public entry points:

- ``colt_drive.prompts``: the structured decision interface used for every
  evaluated model.
- ``colt_drive.inference``: run a Qwen3-VL-style model over a benchmark split.
- ``colt_drive.judge``: map free-form responses to canonical action pairs and
  score them against each sample's accepted set.
"""

from .prompts import BASE_PROMPT, COMPLEX_PROMPT, PROMPT_VARIANTS, SYSTEM_PROMPT

__all__ = [
    "BASE_PROMPT",
    "COMPLEX_PROMPT",
    "PROMPT_VARIANTS",
    "SYSTEM_PROMPT",
]
