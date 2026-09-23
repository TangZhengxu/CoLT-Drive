"""Behaviour regimes for regime-aware routing.

Routine cruising, deliberate maneuvers, and reactive responses are genuinely
different behaviours, so they should not compete for the same adapter weights.
RegMoE gives the router a learned per-regime bias, and the regime label is read
off the supervised action pair rather than annotated separately.

Regimes are used during training only. At inference no regime id is supplied:
the bias embedding is initialised to zero and the gate has to infer the regime
from the input on its own.
"""

from __future__ import annotations

CRUISER = 0
MANEUVERER = 1
REACTOR = 2

NUM_REGIMES = 3

# A large lateral displacement that the navigation goal asks for, rather than
# something in the scene forcing it.
_MANEUVER_LAT_CUES = ("turn", "lane change")

# Reactive cues: an in-lane obstacle or conflict is driving the response.
_REACTOR_LAT_CUES = ("nudge",)
_REACTOR_LON_CUES = (
    "stop",
    "slow",
    "decelerate",
    "brake",
    "reduce speed",
    "yield",
    "adjust",
    "creep",
    "crawl",
    "inch forward",
)


def infer_regime(lon_action: str, lat_action: str) -> int:
    """Derive the behaviour regime from a supervised (longitudinal, lateral) pair.

    Order matters: a navigation maneuver is checked first, so that a turn
    accompanied by braking is still counted as a maneuver rather than a
    reaction.
    """
    lon = (lon_action or "").lower().replace("_", " ").replace("-", " ")
    lat = (lat_action or "").lower().replace("_", " ").replace("-", " ")

    if any(cue in lat for cue in _MANEUVER_LAT_CUES):
        return MANEUVERER
    if any(cue in lat for cue in _REACTOR_LAT_CUES) or any(cue in lon for cue in _REACTOR_LON_CUES):
        return REACTOR
    return CRUISER
