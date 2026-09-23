"""Prompt definitions for the CoLT-Drive benchmark.

Two prompts are available:

  - ``COMPLEX_PROMPT``: the structured decision interface. It walks the model
    through PERCEPTION -> SPATIAL REASONING -> DECISION and asks for a parsable
    longitudinal/lateral action pair at the end. This is the default.
  - ``BASE_PROMPT``: a single open-ended question, with no imposed structure.

Every evaluated model receives the same prompt, so the comparison is fair.

Do not edit these strings. Scores are only comparable across models when the
prompt is byte-identical; mutating a prompt silently invalidates every number
previously measured under it. If you need a different phrasing, add a new
constant instead of changing an existing one.
"""

SYSTEM_PROMPT = "You are a helpful autonomous driving assistant."

BASE_PROMPT = (
    "Task: Based on the front camera image and driving context above, "
    "what action should the ego vehicle take right now? "
    "Describe both the lateral (steering) action and the "
    "longitudinal (speed control) action."
)

COMPLEX_PROMPT = (
    "--- PERCEPTION ---\n"
    "1. What type of road is this? How many lanes are visible?\n"
    "2. Identify ALL lane markings (dashed lines, solid lines, road edges) "
    "you can see, from left to right.\n"
    "3. List every obstacle, object, or hazard visible on or near the road. "
    "For each one, describe WHAT it is.\n\n"
    "--- SPATIAL REASONING (gap and width analysis) ---\n"
    "4. Does the obstacle span the ENTIRE width of the road, from the left "
    "edge to the right edge? (Yes/No)\n"
    "5. If No, look at the space around the obstacle:\n"
    "   a. Is there visible drivable road surface to the LEFT of the obstacle, "
    "wide enough for a car to pass? (Yes/No)\n"
    "   b. Is there visible drivable road surface to the RIGHT of the obstacle, "
    "wide enough for a car to pass? (Yes/No)\n"
    "6. Is the obstacle directly ahead of the ego vehicle (in the ego vehicle's "
    "path)? (Yes/No)\n\n"
    "--- DECISION ---\n"
    "Based on ALL your analysis above, what should the ego vehicle do right now?\n"
    "- Lateral action (steering): describe your action.\n"
    "- Longitudinal action (speed control): describe your action."
)

PROMPT_VARIANTS: dict[str, str] = {
    "complex": COMPLEX_PROMPT,
    "base": BASE_PROMPT,
}
