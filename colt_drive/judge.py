#!/usr/bin/env python3
"""CoLT-Drive decision judge (single file, self-contained).

Scores a driving VLM's free-form responses on the CoLT-Drive counterfactual
obstacle benchmark. Two stages, no external Python dependencies beyond
`requests`:

  1. Decision extraction: pull the final longitudinal/lateral decision out of
     the model's raw response and normalize common phrasings (markdown,
     prompt-echo preambles, compound/hedged lines, directional U-turns,
     prose-only answers, repetition loops).

  2. LLM judge: an LLM maps the extracted decision to canonical action tokens
     and checks membership against the sample's accepted action pairs
     (`data/<split>/<sample_id>/gt.json`). The accepted-pair set is the final
     ground truth; scoring is a plain membership check.

The judge model is queried through any OpenAI-compatible chat-completions
endpoint. Configure it with JUDGE_API_KEY, and optionally JUDGE_API_ENDPOINT
and JUDGE_MODEL.

Usage:
    export JUDGE_API_KEY=...
    python -m colt_drive.judge \
        --results-file  my_model_vfull.json \
        --data-dir      data/vfull \
        --output-dir    results/my_model/vfull

Input `--results-file` schema:
    {"samples": [{"sample_id": "...", "response": "<full model output>"}, ...]}

Outputs into `--output-dir`:
    summary.json                 overall + per-category accuracy
    per_sample.json              per-sample predictions and verdicts
    progress.jsonl               resume log (safe to re-run)
    judge_config.json            cache identity (never contains the API key)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

API_ENDPOINT = os.environ.get(
    "JUDGE_API_ENDPOINT", "https://api.openai.com/v1/chat/completions"
)
DEFAULT_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o")
API_KEY = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""

MAX_RETRIES = 3
RETRY_DELAY_S = 5
REQUEST_TIMEOUT_S = 180
DECISION_INPUT_CHAR_LIMIT = 1500
JUDGE_CACHE_VERSION = 2

# Judging must be as deterministic as the endpoint allows, so temperature=0 is
# the default. Some reasoning endpoints reject it and only accept the default of
# 1.0; those are matched here.
FIXED_TEMPERATURE_JUDGE_RE = re.compile(r"(?:^|/)(?:o[134]\b|gpt-5)", re.IGNORECASE)

# Judges that emit an internal reasoning trace bill it against max_tokens, so
# they need a larger budget to still return the answer.
LARGE_BUDGET_JUDGE_RE = re.compile(r"gemini|deepseek|reasoner|thinking", re.IGNORECASE)

VALID_LON = {"keep_speed", "slow_down", "yield", "creep", "stop", "speed_up"}
VALID_LAT = {"keep_lane", "nudge_left", "nudge_right"}

_progress_lock = threading.Lock()


# ===========================================================================
# Stage 1: decision extraction
# ===========================================================================

SECTION_DECISION_PATTERN = re.compile(
    r"(?:^|\n)\s*"
    r"(?:#{1,6}\s*(?:---\s*)?)?"
    r"(?:---\s*|\*{2}\s*)?"
    r"(?:\*{2}\s*)?"
    r"(?:final\s+)?(?:decision|action\s+summary|recommendation)"
    r"(?:\s*\*{2})?"
    r"\s*[:\-]*\s*\*{0,2}"
    r"\s*(?:\n|$)",
    re.IGNORECASE | re.MULTILINE,
)
INLINE_DECISION_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:final\s+(?:decision|answer)|decision)\s*:",
    re.IGNORECASE | re.MULTILINE,
)
ACTION_LINE_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:longitudinal|lateral)\s*:",
    re.IGNORECASE,
)
_LEADING_MARKUP = re.compile(r"^[\s:*\-]+")

PROMPT_ECHO_PATTERNS = [
    re.compile(r"^\s*based on all your analysis above[^\n]*\n", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*based on all the analys(?:e?s|is(?:\s+above)?)[^\n]*\n", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*based on all my analysis above[^\n]*\n", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*based on the analys(?:e?s)?:?\s*\n", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*based on all analyses?:?\s*\n", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*based on all analyses?[^\n]*\n", re.IGNORECASE | re.MULTILINE),
]

HEDGE_MARKERS = [
    "if necessary", "if required", "if needed", "as needed", "as required",
    "if possible", "if the gap", "if appropriate", "alternatively",
    "potentially", "either", " or ", "/", "slightly",
]

COMMITTED_TRAILING_PATTERNS = [
    re.compile(r"\b(?:vehicle\s+)?must\s+(?:nudge|steer|turn|shift|change\s+lanes|move)\b", re.IGNORECASE),
    re.compile(r"\b(?:ego\s+vehicle\s+|vehicle\s+)?should\s+nudge\b", re.IGNORECASE),
    re.compile(r"\bnudge\s+(?:to\s+the\s+)?(?:left|right)\s+(?:through|out\s+of|across)\b", re.IGNORECASE),
    re.compile(r"\bnudge\s+through\s+the\s+gap\b", re.IGNORECASE),
    re.compile(r"\b(?:final|ultimate)\s+(?:action|decision)[^:]*:\s*nudge\s+(?:left|right)\b", re.IGNORECASE),
]

LAT_ACTION_RE = re.compile(
    r"\b(?:keep\s+(?:the\s+)?lane|stay\s+in\s+(?:the\s+)?lane|nudge|steer|"
    r"veer|shift|swerve|merge|lane\s+change|change\s+lanes|turn\s+(?:left|right)|"
    r"maintain\s+(?:current\s+)?(?:trajectory|position|course|lane))\b",
    re.IGNORECASE,
)
LON_ACTION_RE = re.compile(
    r"\b(?:keep\s+speed|maintain\s+(?:current\s+)?speed|slow\s+down|decelerate|"
    r"brake|reduce\s+speed|stop|halt|yield|give\s+way|creep|crawl|inch\s+forward|"
    r"speed\s+up|accelerate)\b",
    re.IGNORECASE,
)


def _clean_leading_markup(text: str) -> str:
    return _LEADING_MARKUP.sub("", text).strip()


def _extract_decision_segment(raw_response: str) -> str:
    if not raw_response or not isinstance(raw_response, str):
        return ""
    section_match = SECTION_DECISION_PATTERN.search(raw_response)
    if section_match:
        body = raw_response[section_match.end():]
        return _clean_leading_markup(body)
    inline_matches = list(INLINE_DECISION_PATTERN.finditer(raw_response))
    if inline_matches:
        last = inline_matches[-1]
        body = raw_response[last.end():]
        return _clean_leading_markup(body)
    action_matches = list(ACTION_LINE_PATTERN.finditer(raw_response))
    if action_matches:
        last = action_matches[-1]
        start = last.start()
        for prev in reversed(action_matches[:-1]):
            between = raw_response[prev.end():start]
            if "\n\n" in between:
                break
            if re.search(r"\n\s*(?:#{1,6}\s+|\*\*)", between):
                break
            start = prev.start()
        return _clean_leading_markup(raw_response[start:])
    return ""


def _strip_markdown(text: str) -> str:
    text = re.sub(r"\*\*+", "", text)
    text = re.sub(r"^\s*[-*\u2022]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?im)\blateral\s+action\s*(?:\([^)]*\))?\s*:", "Lateral:", text)
    text = re.sub(r"(?im)\blongitudinal\s+action\s*(?:\([^)]*\))?\s*:", "Longitudinal:", text)
    text = re.sub(r"(?im)\bsteering\s+action\s*:", "Lateral:", text)
    text = re.sub(r"(?im)\bspeed\s+control\s+action\s*:", "Longitudinal:", text)
    text = re.sub(r"(?im)\bspeed\s+action\s*:", "Longitudinal:", text)
    return text


def _strip_prompt_echo(text: str) -> str:
    for pat in PROMPT_ECHO_PATTERNS:
        text = pat.sub("", text)
    return text


def _truncate_repetition(text: str) -> str:
    if not text or len(text) < 300:
        return text
    lines = text.split("\n")
    for i in range(len(lines) - 3):
        line = lines[i].strip()
        if (
            line
            and len(line) < 100
            and lines[i + 1].strip() == line
            and lines[i + 2].strip() == line
            and lines[i + 3].strip() == line
        ):
            return "\n".join(lines[:i]).strip()
    if len(text) > 2000:
        chunks: dict[str, int] = {}
        for i in range(0, len(text) - 30, 30):
            c = text[i:i + 30]
            chunks[c] = chunks.get(c, 0) + 1
        for chunk, count in chunks.items():
            if count >= 8:
                idx = text.find(chunk)
                if idx > 100:
                    return text[:idx].strip()
    return text


def _split_compound_clause(line_content: str) -> str:
    if not line_content:
        return line_content
    match = re.search(r"[,;]|\.\s+(?=[A-Z])|\.\s*$|\bsince\b|\bhowever\b", line_content, re.IGNORECASE)
    if not match:
        return line_content.strip()
    leading = line_content[:match.start()].strip()
    trailing = line_content[match.end():].strip()
    if len(leading) < 5:
        return line_content.strip()
    if not trailing or len(trailing) < 5:
        return leading

    trailing_lower = trailing.lower()
    has_hedge = any(m in trailing_lower for m in HEDGE_MARKERS)
    has_committed = any(p.search(trailing) for p in COMMITTED_TRAILING_PATTERNS)

    return trailing if has_committed and not has_hedge else leading


def _uturn_directional(line_content: str) -> str:
    line_content = re.sub(r"\bu[-\s]?turn\s+(?:to\s+)?(?:the\s+)?left\b", "nudge left", line_content, flags=re.IGNORECASE)
    line_content = re.sub(r"\bu[-\s]?turn\s+(?:to\s+)?(?:the\s+)?right\b", "nudge right", line_content, flags=re.IGNORECASE)
    return line_content


def _per_line_normalize(text: str) -> str:
    out_lines: list[str] = []
    for line in text.split("\n"):
        m_lat = re.match(r"\s*Lateral\s*:\s*(.*)", line, re.IGNORECASE)
        m_lon = re.match(r"\s*Longitudinal\s*:\s*(.*)", line, re.IGNORECASE)
        if m_lat:
            content = m_lat.group(1)
            content = _uturn_directional(content)
            chosen = _split_compound_clause(content)
            out_lines.append(f"Lateral: {chosen}")
        elif m_lon:
            content = m_lon.group(1)
            chosen = _split_compound_clause(content)
            out_lines.append(f"Longitudinal: {chosen}")
        else:
            out_lines.append(line)

    return "\n".join(out_lines)


def _has_axis_line(text: str) -> bool:
    return bool(
        re.search(r"(?im)^\s*Lateral\s*:", text)
        or re.search(r"(?im)^\s*Longitudinal\s*:", text)
    )


def _prose_fallback(text: str) -> str:
    additions: list[str] = []
    text_lower = text.lower()

    if re.search(r"\bshould\s+slow\s+down\b", text_lower):
        additions.append("Longitudinal: slow down")
    elif re.search(r"\bshould\s+reduce\s+speed\b", text_lower):
        additions.append("Longitudinal: slow down")
    elif re.search(r"\bshould\s+(?:come\s+to\s+a\s+(?:complete\s+)?)?stop\b", text_lower):
        additions.append("Longitudinal: stop")
    elif re.search(r"\bcome\s+to\s+a\s+(?:complete\s+|full\s+)?stop\b", text_lower):
        additions.append("Longitudinal: stop")
    elif re.search(r"\bshould\s+yield\b", text_lower):
        additions.append("Longitudinal: yield")
    elif re.search(r"\bshould\s+maintain\s+(?:its\s+)?current\s+speed\b", text_lower):
        additions.append("Longitudinal: keep speed")

    if re.search(r"\bshould\s+maintain\s+(?:its\s+)?current\s+(?:position|trajectory|course)\b", text_lower):
        additions.append("Lateral: keep lane")
    elif re.search(r"\bmaintain\s+(?:its\s+)?current\s+(?:position|trajectory|course)\b", text_lower):
        additions.append("Lateral: keep lane")
    elif re.search(r"\bshould\s+keep\s+lane\b", text_lower):
        additions.append("Lateral: keep lane")
    elif re.search(r"\bshould\s+steer\s+(?:slightly\s+)?(?:to\s+the\s+)?left\b", text_lower):
        additions.append("Lateral: nudge left")
    elif re.search(r"\bshould\s+steer\s+(?:slightly\s+)?(?:to\s+the\s+)?right\b", text_lower):
        additions.append("Lateral: nudge right")
    elif re.search(r"\bshould\s+nudge\s+(?:to\s+the\s+)?left\b", text_lower):
        additions.append("Lateral: nudge left")
    elif re.search(r"\bshould\s+nudge\s+(?:to\s+the\s+)?right\b", text_lower):
        additions.append("Lateral: nudge right")

    if additions:
        return text.strip() + "\n\n" + "\n".join(additions)
    return text


TAIL_ACTION_PATTERNS: list[tuple[re.Pattern[str], tuple[str, str]]] = [
    (re.compile(r"\bcome\s+to\s+a\s+(?:complete|full)?\s*stop\b", re.IGNORECASE), ("Longitudinal", "stop")),
    (re.compile(r"\bstop\s+immediately\b", re.IGNORECASE), ("Longitudinal", "stop")),
    (re.compile(r"\bslow\s+down\b", re.IGNORECASE), ("Longitudinal", "slow down")),
    (re.compile(r"\breduce\s+speed\b", re.IGNORECASE), ("Longitudinal", "slow down")),
    (re.compile(r"\bmaintain\s+(?:its\s+)?(?:current\s+)?speed\b", re.IGNORECASE), ("Longitudinal", "keep speed")),
    (re.compile(r"\byield\s+(?:to|until|before)\b", re.IGNORECASE), ("Longitudinal", "yield")),
    (re.compile(r"\bkeep\s+lane\b", re.IGNORECASE), ("Lateral", "keep lane")),
    (re.compile(r"\bnudge\s+left\b", re.IGNORECASE), ("Lateral", "nudge left")),
    (re.compile(r"\bnudge\s+right\b", re.IGNORECASE), ("Lateral", "nudge right")),
    (re.compile(r"\bmaintain\s+(?:its\s+)?(?:current\s+)?(?:trajectory|position|course)\b", re.IGNORECASE), ("Lateral", "keep lane")),
]


def _extract_tail_actions(raw_response: str) -> str:
    if not raw_response:
        return ""
    paragraphs = re.split(r"\n\s*\n", raw_response.strip())
    tail = "\n\n".join(paragraphs[-3:]) if len(paragraphs) >= 3 else raw_response

    found_lat: tuple[str, str] | None = None
    found_lon: tuple[str, str] | None = None
    for pat, (axis, canonical) in TAIL_ACTION_PATTERNS:
        m = pat.search(tail)
        if m:
            window = tail[max(0, m.start() - 30):min(len(tail), m.end() + 30)].lower()
            if any(h in window for h in ["if necessary", "if needed", "if required", "alternatively", " or "]):
                continue
            if axis == "Lateral" and found_lat is None:
                found_lat = (axis, canonical)
            elif axis == "Longitudinal" and found_lon is None:
                found_lon = (axis, canonical)

    if found_lat or found_lon:
        lines: list[str] = []
        if found_lat:
            lines.append(f"{found_lat[0]}: {found_lat[1]}")
        if found_lon:
            lines.append(f"{found_lon[0]}: {found_lon[1]}")
        return "\n".join(lines)
    return ""


def normalize_decision_block(text: str) -> str:
    if not text:
        return text
    text = _strip_markdown(text)
    text = _strip_prompt_echo(text)
    text = _truncate_repetition(text)
    text = _per_line_normalize(text)
    if not _has_axis_line(text):
        text = _prose_fallback(text)
    return text.strip()


def extract_decision(raw_response: str, variant: str = "complex") -> str:
    """Return the normalized decision text for one raw model response."""
    if variant == "base":
        stripped = (raw_response or "").strip()
        return normalize_decision_block(stripped) if stripped else ""

    decision_text = _extract_decision_segment(raw_response)
    if not decision_text:
        decision_text = _extract_tail_actions(raw_response)
    if not decision_text:
        return ""
    return normalize_decision_block(decision_text)


# ===========================================================================
# Stage 2: LLM judge
# ===========================================================================

JUDGE_PROMPT_TEMPLATE = """\
You are a strict, literal information extractor. Your task is to read a short
DECISION text from a vision-language driving model, map its phrasing to
canonical action tokens, and check membership in the ground-truth pair list.

## CANONICAL TOKEN TABLE (READ THIS FIRST AND USE IT LITERALLY)

You MUST map natural language to exactly one of these tokens:

### Lateral (steering) -> one of: keep_lane | nudge_left | nudge_right | none

- **keep_lane**: "keep straight", "stay in current lane", "keep lane",
  "keep lane centered", "no lateral shift", "follow the current lane",
  "maintain lane", "maintain current trajectory", "maintain its current
  trajectory", "maintain its current position", "maintain its current course",
  "maintain current lateral position", "hold the center of the lane",
  "return to original lateral position", "no steering adjustment",
  "drive straight ahead", "continue straight",
  "the ego vehicle should maintain (its) current position/trajectory/course".
- **nudge_left**: "steer left", "veer left", "shift left", "drift left",
  "swerve left", "turn left to avoid", "avoid to the left",
  "nudge left", "nudge to the left", "nudge slightly left",
  "merge left into the lane", "lane change left", "change lanes to the left",
  "U-turn left", "U-turn to the left", "steer slightly to the left".
- **nudge_right**: same family, right-side.

### Longitudinal (speed) -> one of: keep_speed | slow_down | stop | speed_up | yield | creep | none

- **slow_down**: "decelerate", "brake gently", "reduce speed",
  "slow down", "lower speed", "ease off",
  "adjust speed", "adjust speed for road events", "adjust speed for caution",
  "slow down and prepare to stop", "slow down and maintain a safe distance",
  "should maintain a safe distance" (implies slowing),
  "keep a safe distance" (implies slowing),
  "proceed cautiously", "proceed with caution",
  "accelerate cautiously" (cautiously dominates accelerate).
- **stop**: "halt", "full stop", "bring to a stop",
  "come to a complete stop", "apply full braking", "stop immediately",
  "stop and wait", "stop and wait until the obstacle is cleared",
  "remain stationary".
- **keep_speed**: "maintain speed", "maintain current speed",
  "continue at current speed", "continue at a steady rate",
  "maintain a steady pace", "proceed", "keep speed", "maintain velocity",
  "no significant change in speed",
  "the ego vehicle should maintain its current speed".
- **speed_up**: "accelerate", "accelerate to target speed",
  "continue accelerating at a steady rate", "speed up".
- **yield**: "give way", "let the obstacle pass",
  "wait for the obstacle to pass", "wait for the obstacle to clear",
  "let them clear first", "yield", "yield and keep a safe distance".
- **creep**: "inch forward", "crawl forward", "creep", "move very slowly".

## EXTRACTION & MATCH RULES

1. NO HEDGING: If the decision recommends multiple contradictory plans without
   committing (e.g. "steer left, or stay in lane if too narrow", "slow down or
   keep speed"), set the affected axis to "none". The default-fill rule (7)
   below applies only when an axis is genuinely missing, not when it is
   hedging-ambiguous.

2. MISSING ACTIONS: If the decision text is empty, only describes the scene,
   or otherwise does not explicitly recommend a steering or speed-control
   action, set that axis to "none". Do NOT default to keep_lane / keep_speed
   inside the LLM judgment - the wrapper handles that downstream via rule 7.

3. COMPOUND PHRASES (leading-verb wins for hedged trailings, but trailing-
   verb wins for clearly-committed independent sentences):
   - "Lateral: keep lane, keep a safe distance" -> keep_lane (the trailing
     "keep a safe distance" is a safety qualifier).
   - "Lateral: keep lane, nudge slightly left if necessary" -> keep_lane
     ("if necessary" hedges the trailing).
   - "Lateral: keep lane. ... ego vehicle should nudge through the gap" ->
     nudge (trailing is independently committed without hedging).
   You typically receive a single canonical verb per axis.

4. LANE-CHANGE / U-TURN PHRASES: "change lanes left", "lane change left",
   "switch to the left lane", "merge left", "turn left to avoid", and
   "U-turn left" / "U-turn to the left" are all mapped to nudge_left.
   Right-hand counterparts to nudge_right. Bare "U-turn" without direction
   stays "none".

5. UNRECOVERABLE OUTPUT: If the decision text is in a language other than
   English, contains no recognizable canonical action vocabulary, or is
   garbled / repetitive, set both pred_lat and pred_lon to "none".

6. NORMALIZE OUTPUT: Always lowercase, always one of the canonical tokens
   or "none".

7. MISSING-AXIS DEFAULT FILL (LLM side): If exactly one axis is determinable
   (after synonym mapping) and the other axis has NO action verb at all on
   its line (the line is genuinely missing or contains only non-action
   prose), fill the missing axis with the canonical no-op default:
     - pred_lat missing -> pred_lat = "keep_lane"
     - pred_lon missing -> pred_lon = "keep_speed"
   This rule does NOT apply when the missing axis exists but is "none" due
   to hedging or contradiction (rule 1). The wrapper will also apply this
   fill downstream as a safety net.

## ACCEPTABLE GROUND TRUTH PAIRS (any one pair is correct)
{pairs_text}

## FEW-SHOT REFERENCE

### Example A (clear conversational decision)
- Decision Text: "The vehicle should brake gently to maintain safety, and veer slightly left to bypass the pothole."
- Output: {{"pred_lat": "nudge_left", "pred_lon": "slow_down", "pair_match": true, "reasoning": "veer slightly left -> nudge_left; brake gently -> slow_down."}}

### Example B (hedging on lateral, commits longitudinal)
- Decision Text: "Steer right if the gap is wide enough, otherwise stay centered in the current lane. Keep current speed."
- Output: {{"pred_lat": "none", "pred_lon": "keep_speed", "pair_match": false, "reasoning": "Hedges between nudge_right and keep_lane; lateral is none."}}

### Example C (missing lateral, only filler)
- Decision Text: "Longitudinal: slow down. Lateral: proceed with caution."
- Output: {{"pred_lat": "none", "pred_lon": "slow_down", "pair_match": false, "reasoning": "'proceed with caution' is not a steering recommendation; lateral is none."}}

### Example D (terse two-line short answer)
- Decision Text: "Longitudinal: keep speed\\nLateral: turn left"
- Output: {{"pred_lat": "nudge_left", "pred_lon": "keep_speed", "pair_match": false, "reasoning": "turn left -> nudge_left; keep speed -> keep_speed."}}

### Example E (compound phrase with safety qualifiers)
- Decision Text: "Longitudinal: decelerate to keep a safe distance\\nLateral: keep lane, keep a safe distance"
- Output: {{"pred_lat": "keep_lane", "pred_lon": "slow_down", "pair_match": false, "reasoning": "Leading lateral verb is 'keep lane'; trailing 'keep a safe distance' is a safety qualifier. 'decelerate' -> slow_down."}}

### Example F (prompt-echo preamble)
- Decision Text: "Based on ALL your analysis above, what should the ego vehicle do right now?\\n- Lateral action (steering): keep lane\\n- Longitudinal action (speed control): slow down due to caution while navigating the hole"
- Output: {{"pred_lat": "keep_lane", "pred_lon": "slow_down", "pair_match": false, "reasoning": "Ignored prompt-echo preamble. 'keep lane' -> keep_lane; 'slow down' -> slow_down."}}

### Example G (prompt-echo + markdown + committed trailing)
- Decision Text: "Lateral: Keep lane. ... vehicle must nudge slightly out of the lane if necessary.\\nLongitudinal: Slow down."
- Output: {{"pred_lat": "keep_lane", "pred_lon": "slow_down", "pair_match": false, "reasoning": "Leading 'Keep lane' wins because 'if necessary' hedges the trailing 'must nudge'. slow down -> slow_down."}}

### Example H (directional U-turn)
- Decision Text: "Lateral: U-turn left\\nLongitudinal: slow down"
- Output: {{"pred_lat": "nudge_left", "pred_lon": "slow_down", "pair_match": false, "reasoning": "'U-turn left' aliases nudge_left per rule 4; slow down -> slow_down."}}

### Example I (adjust speed for road events -> slow_down)
- Decision Text: "Lateral: keep lane\\nLongitudinal: adjust speed for road events"
- Output: {{"pred_lat": "keep_lane", "pred_lon": "slow_down", "pair_match": false, "reasoning": "'adjust speed for road events' is in the slow_down synonym list; keep lane -> keep_lane."}}

### Example J (genuinely missing lateral -> fill default)
- Decision Text: "Longitudinal: stop and wait until the obstacle is cleared."
- Output: {{"pred_lat": "keep_lane", "pred_lon": "stop", "pair_match": false, "reasoning": "Lateral line is genuinely missing (no steering verb anywhere); rule 7 fills keep_lane. 'stop and wait' -> stop."}}

## Decision Segment to Evaluate
{response}

Output ONLY valid JSON with no markdown formatting:
{{"pred_lat": "keep_lane|nudge_left|nudge_right|none", "pred_lon": "keep_speed|slow_down|stop|speed_up|yield|creep|none", "pair_match": true|false, "reasoning": "<1 short sentence>"}}"""


def _normalize_axis(value: Any, valid: set[str]) -> str:
    if value is None:
        return "none"
    text = str(value).strip().lower()
    if text in {"", "none", "null", "n/a", "na", "unknown"}:
        return "none"
    return text if text in valid else "none"


def _extract_json_object(content: str) -> dict[str, Any] | None:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```\w*\n?", "", content)
        content = re.sub(r"\n?```$", "", content).strip()
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _build_payload(prompt: str, judge_model: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
    }
    payload["temperature"] = 1.0 if FIXED_TEMPERATURE_JUDGE_RE.search(judge_model) else 0.0
    payload["max_tokens"] = 4096 if LARGE_BUDGET_JUDGE_RE.search(judge_model) else 2048
    return payload


def _empty_match(reasoning: str) -> dict[str, Any]:
    return {"pred_lat": "none", "pred_lon": "none", "pair_match": False, "reasoning": reasoning}


def call_judge(decision_text: str, acceptable_pairs: list[dict], judge_model: str) -> dict[str, Any] | None:
    decision_text = (decision_text or "").strip()
    if not decision_text:
        return _empty_match("Decision segment was empty (no DECISION section detected).")

    pairs_text = "\n".join(
        f"  {i+1}. Longitudinal: {p['lon_action']}, Lateral: {p['lat_action']}"
        for i, p in enumerate(acceptable_pairs)
    )
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        pairs_text=pairs_text,
        response=decision_text[:DECISION_INPUT_CHAR_LIMIT],
    )
    payload = _build_payload(prompt, judge_model)
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_S)
            if resp.status_code == 429:
                time.sleep(RETRY_DELAY_S * (2 ** attempt))
                continue
            if resp.status_code in {400, 401, 403}:
                raise RuntimeError(
                    f"Judge API rejected the request (HTTP {resp.status_code}). "
                    "Check JUDGE_API_ENDPOINT, JUDGE_MODEL, and JUDGE_API_KEY."
                )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            if not content:
                reasoning_content = (msg.get("reasoning_content") or "").strip()
                m = re.search(r"\{[^{}]*pred_lat[^{}]*\}", reasoning_content, flags=re.DOTALL)
                if m:
                    content = m.group(0)
                else:
                    time.sleep(RETRY_DELAY_S * (2 ** attempt))
                    continue
            parsed = _extract_json_object(content)
            if parsed is None or "pred_lat" not in parsed or "pred_lon" not in parsed:
                time.sleep(RETRY_DELAY_S * (2 ** attempt))
                continue
            return parsed
        except (json.JSONDecodeError, requests.exceptions.RequestException, KeyError):
            time.sleep(RETRY_DELAY_S * (2 ** attempt))
        except RuntimeError:
            raise
        except Exception:
            return None
    return None


# ===========================================================================
# Ground truth + scoring
# ===========================================================================

def load_gt(data_dir: Path, sample_id: str) -> dict[str, Any]:
    p = data_dir / sample_id / "gt.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_meta(data_dir: Path, sample_id: str) -> dict[str, Any]:
    p = data_dir / sample_id / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def resolve_pairs(data_dir: Path, sample_id: str) -> list[dict[str, str]]:
    """Load the authoritative accepted pairs from ``gt.json``."""
    candidate = load_gt(data_dir, sample_id).get("acceptable_pairs")
    if not isinstance(candidate, list) or not candidate:
        raise ValueError(f"Missing or empty acceptable_pairs for {sample_id}")
    pairs = [
        {"lon_action": p.get("lon_action", ""), "lat_action": p.get("lat_action", "")}
        for p in candidate
        if isinstance(p, dict)
    ]
    if not pairs or any(
        p["lon_action"] not in VALID_LON or p["lat_action"] not in VALID_LAT
        for p in pairs
    ):
        raise ValueError(f"Invalid acceptable_pairs for {sample_id}")
    return pairs


def python_pair_match(pred_lon: str, pred_lat: str, pairs: list[dict[str, str]]) -> bool:
    if pred_lon == "none" or pred_lat == "none":
        return False
    return any(p.get("lon_action") == pred_lon and p.get("lat_action") == pred_lat for p in pairs)


def _axis_is_genuinely_missing(decision_text: str, axis: str) -> bool:
    """Return true only when the response contains no action for that axis."""
    if axis == "lat":
        if re.search(r"(?im)^\s*Lateral\s*:", decision_text):
            return False
        return LAT_ACTION_RE.search(decision_text) is None
    if re.search(r"(?im)^\s*Longitudinal\s*:", decision_text):
        return False
    return LON_ACTION_RE.search(decision_text) is None


def apply_missing_axis_fill(
    pred_lat: str, pred_lon: str, decision_text: str
) -> tuple[str, str, list[str]]:
    filled: list[str] = []
    if (
        pred_lat == "none"
        and pred_lon != "none"
        and _axis_is_genuinely_missing(decision_text, "lat")
    ):
        pred_lat = "keep_lane"
        filled.append("lat")
    if (
        pred_lon == "none"
        and pred_lat != "none"
        and _axis_is_genuinely_missing(decision_text, "lon")
    ):
        pred_lon = "keep_speed"
        filled.append("lon")
    return pred_lat, pred_lon, filled


def load_progress(path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = entry.get("sample_id")
            if sid:
                done[sid] = entry
    return done


def append_progress(path: Path, entry: dict[str, Any]) -> None:
    with _progress_lock:
        with path.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _gt_fingerprint(data_dir: Path, sample_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for sid in sample_ids:
        path = data_dir / sid / "gt.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing ground truth: {path}")
        digest.update(sid.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _prepare_progress_cache(
    output_dir: Path,
    progress_path: Path,
    run_config: dict[str, Any],
) -> None:
    config_path = output_dir / "judge_config.json"
    existing = None
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            existing = None
    if progress_path.exists() and existing != run_config:
        print("  Judge configuration changed; clearing stale progress cache.")
        progress_path.unlink()
    config_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=False))


def judge_one(sample: dict[str, Any], idx: int, total: int, data_dir: Path, judge_model: str, variant: str) -> dict[str, Any]:
    sid = sample["sample_id"]
    decision_text = extract_decision(str(sample.get("response", "") or ""), variant)
    decision_found = bool(decision_text.strip())
    meta = load_meta(data_dir, sid)
    pairs = resolve_pairs(data_dir, sid)

    t0 = time.time()
    if not decision_found:
        result: dict[str, Any] | None = _empty_match("No DECISION section in raw response.")
        judge_failed = False
    else:
        result = call_judge(decision_text, pairs, judge_model)
        judge_failed = result is None
        if judge_failed:
            result = _empty_match("JUDGE_FAILED")
    elapsed = time.time() - t0

    pred_lon_raw = _normalize_axis(result.get("pred_lon"), VALID_LON)
    pred_lat_raw = _normalize_axis(result.get("pred_lat"), VALID_LAT)
    pred_lat, pred_lon, filled_axes = apply_missing_axis_fill(
        pred_lat_raw, pred_lon_raw, decision_text
    )
    pair_match = python_pair_match(pred_lon, pred_lat, pairs)

    entry = {
        "sample_id": sid,
        "obstacle_type": meta.get("obstacle_type", ""),
        "obstacle_category": meta.get("obstacle_category", ""),
        "position": meta.get("position", ""),
        "scene_id": meta.get("scene_id", ""),
        "version": meta.get("version", ""),
        "gt_pairs": [[p["lon_action"], p["lat_action"]] for p in pairs],
        "pred_lat": pred_lat,
        "pred_lon": pred_lon,
        "pred_lat_pre_fill": pred_lat_raw,
        "pred_lon_pre_fill": pred_lon_raw,
        "missing_axis_filled": filled_axes,
        "pair_match": pair_match,
        "judge_reasoning": str(result.get("reasoning", ""))[:1000],
        "judge_failed": judge_failed,
        "decision_found": decision_found,
        "decision_snippet": decision_text[:300],
        "judge_time_s": round(elapsed, 2),
    }

    if not decision_found:
        tag = "EMPTY"
    elif judge_failed:
        tag = "FAIL"
    elif pair_match:
        tag = "MATCH"
    elif pred_lon == "none" or pred_lat == "none":
        tag = "NONE"
    else:
        tag = "MISS"
    print(f"  [{idx+1}/{total}] {sid:<45} | {elapsed:5.1f}s | pred=({pred_lon},{pred_lat}) | {tag}", flush=True)
    return entry


def evaluate(
    results_file: Path,
    data_dir: Path,
    output_dir: Path,
    workers: int,
    judge_model: str,
    variant: str,
    allow_judge_failures: bool = False,
) -> dict[str, Any]:
    data = json.loads(results_file.read_text())
    if not isinstance(data, dict):
        raise ValueError("results-file root must be a JSON object")
    samples = data.get("samples", [])
    if not isinstance(samples, list) or not samples:
        raise ValueError("results-file must contain a non-empty samples list")
    sample_ids = [
        s.get("sample_id") if isinstance(s, dict) else None for s in samples
    ]
    if any(not sid for sid in sample_ids):
        raise ValueError("Every sample must be an object with a non-empty sample_id")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate sample_id values in results-file")
    if variant == "auto":
        recorded_summary = data.get("summary")
        recorded_prompt = (
            recorded_summary.get("prompt", "complex")
            if isinstance(recorded_summary, dict)
            else "complex"
        )
        variant = recorded_prompt if recorded_prompt in {"complex", "base"} else "complex"
    for sid in sample_ids:
        resolve_pairs(data_dir, sid)
    print(f"\nCoLT-Drive judge: {results_file.name} ({len(samples)} samples, model={judge_model}, workers={workers})")
    print(f"  Variant: {variant}")

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    stat = results_file.stat()
    run_config = {
        "cache_version": JUDGE_CACHE_VERSION,
        "results_file": str(results_file.resolve()),
        "results_size": stat.st_size,
        "results_mtime_ns": stat.st_mtime_ns,
        "results_fingerprint": hashlib.sha256(results_file.read_bytes()).hexdigest(),
        "data_dir": str(data_dir.resolve()),
        "gt_fingerprint": _gt_fingerprint(data_dir, sample_ids),
        "judge_model": judge_model,
        "variant": variant,
        "endpoint_hash": hashlib.sha256(API_ENDPOINT.encode()).hexdigest(),
    }
    _prepare_progress_cache(output_dir, progress_path, run_config)

    done = load_progress(progress_path)
    print(f"  {len(done)} already judged")

    results: list[dict[str, Any]] = []
    todo: list[tuple[dict[str, Any], int, int]] = []
    for i, s in enumerate(samples):
        sid = s["sample_id"]
        if sid in done and not done[sid].get("judge_failed"):
            results.append(done[sid])
        else:
            todo.append((s, i, len(samples)))
    print(f"  {len(todo)} to judge")

    if workers > 1 and todo:
        # Probe the endpoint with one sample before launching a large request
        # burst. Authentication/configuration failures then stop after one call.
        first_sample, first_idx, first_total = todo.pop(0)
        first_entry = judge_one(
            first_sample, first_idx, first_total, data_dir, judge_model, variant
        )
        results.append(first_entry)
        append_progress(progress_path, first_entry)
        if first_entry.get("judge_failed") and not allow_judge_failures:
            raise RuntimeError(
                "Initial judge call failed; refusing to launch concurrent requests."
            )

    if workers <= 1:
        for s, i, total in todo:
            entry = judge_one(s, i, total, data_dir, judge_model, variant)
            results.append(entry)
            append_progress(progress_path, entry)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(judge_one, s, i, total, data_dir, judge_model, variant): s["sample_id"]
                for s, i, total in todo
            }
            for fut in as_completed(futures):
                entry = fut.result()
                results.append(entry)
                append_progress(progress_path, entry)

    order = {s["sample_id"]: i for i, s in enumerate(samples)}
    results.sort(key=lambda r: order.get(r.get("sample_id"), 10 ** 9))

    n = len(results)
    n_match = sum(1 for r in results if r.get("pair_match"))
    n_judge_failed = sum(1 for r in results if r.get("judge_failed"))
    n_missing_decision = sum(1 for r in results if not r.get("decision_found", True))

    cat_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    pos_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in results:
        m = int(bool(r.get("pair_match")))
        cat = r.get("obstacle_category") or "unknown"
        pos = r.get("position") or "unknown"
        cat_stats[cat][1] += 1
        cat_stats[cat][0] += m
        pos_stats[pos][1] += 1
        pos_stats[pos][0] += m

    summary = {
        "results_file": str(results_file),
        "data_dir": str(data_dir),
        "judge_model": judge_model,
        "variant": variant,
        "cache_version": JUDGE_CACHE_VERSION,
        "total": n,
        "pair_accuracy": round(n_match / max(n, 1) * 100, 2),
        "correct": n_match,
        "judge_failed": n_judge_failed,
        "missing_decision_count": n_missing_decision,
        "category_breakdown": {
            cat: {"correct": v[0], "total": v[1], "accuracy": round(v[0] / max(v[1], 1) * 100, 2)}
            for cat, v in sorted(cat_stats.items())
        },
        "position_breakdown": {
            pos: {"correct": v[0], "total": v[1], "accuracy": round(v[0] / max(v[1], 1) * 100, 2)}
            for pos, v in sorted(pos_stats.items())
        },
    }

    (output_dir / "per_sample.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n  Overall pair_accuracy = {summary['pair_accuracy']}% ({n_match}/{n})")
    print(f"  judge_failed={n_judge_failed}  missing_decision={n_missing_decision}")
    print(f"  Saved: {output_dir / 'summary.json'}")
    if n_judge_failed and not allow_judge_failures:
        raise RuntimeError(
            f"{n_judge_failed} judge calls failed. Results were saved for inspection; "
            "rerun after fixing the endpoint or pass --allow-judge-failures."
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="CoLT-Drive decision judge")
    parser.add_argument("--results-file", type=Path, required=True, help="Model raw output JSON with samples[].response")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to data/<split>/ (holds <sample_id>/gt.json, meta.json)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16, help="Concurrent judge API requests")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--variant", type=str, default="auto", choices=["auto", "complex", "base"],
                        help="'auto' reads the prompt from inference output; "
                             "'complex' extracts DECISION; 'base' uses the whole response.")
    parser.add_argument("--allow-judge-failures", action="store_true",
                        help="Exit successfully even when one or more API calls fail")
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: set JUDGE_API_KEY (or OPENAI_API_KEY)", file=sys.stderr)
        sys.exit(1)
    if not args.results_file.exists():
        print(f"ERROR: missing {args.results_file}", file=sys.stderr)
        sys.exit(1)
    if not args.data_dir.exists():
        print(f"ERROR: missing {args.data_dir}", file=sys.stderr)
        sys.exit(1)
    if args.workers < 1:
        print("ERROR: --workers must be at least 1", file=sys.stderr)
        sys.exit(1)

    evaluate(
        args.results_file,
        args.data_dir,
        args.output_dir,
        args.workers,
        args.judge_model,
        args.variant,
        allow_judge_failures=args.allow_judge_failures,
    )


if __name__ == "__main__":
    main()
