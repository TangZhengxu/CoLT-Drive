"""Minimal JSONL-backed dataset for RegMoE training.

Bring your own driving data. Each line of the manifest is one keyframe:

    {"image": "images/000001.jpg",
     "prompt": "Ego speed 8.2 m/s ... Navigation: go straight ... <question>",
     "target": "Longitudinal: slow down\\nLateral: keep lane"}

Fields:
    image   Path to the front camera frame, absolute or relative to ``root``.
    prompt  The full text context shown to the model: ego history, navigation
            command, and the question. Build it however your data requires; the
            collator does not interpret it.
    target  Supervision string. Loss is computed on these tokens only.
    regime  Optional int in {0, 1, 2}. Omit it and the regime is derived from
            ``target`` via ``kpa.regimes.infer_regime``.

This loader exists so the training code is runnable on any driving dataset that
can be expressed in the schema above.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from torch.utils.data import Dataset

from .regimes import infer_regime

_LON_RE = re.compile(r"longitudinal\s*:\s*(.+)", re.IGNORECASE)
_LAT_RE = re.compile(r"lateral\s*:\s*(.+)", re.IGNORECASE)


def parse_action_pair(target: str) -> tuple[str, str]:
    """Pull the (longitudinal, lateral) actions out of a target string."""
    lon = _LON_RE.search(target or "")
    lat = _LAT_RE.search(target or "")
    return (
        lon.group(1).strip() if lon else "",
        lat.group(1).strip() if lat else "",
    )


class DrivingDecisionDataset(Dataset):
    """Keyframe decision samples read from a JSONL manifest."""

    def __init__(
        self,
        manifest: str | Path,
        root: Optional[str | Path] = None,
        max_samples: Optional[int] = None,
    ):
        if max_samples is not None and max_samples < 1:
            raise ValueError("max_samples must be at least 1")
        self.root = Path(root) if root else Path(manifest).parent
        self.records: list[dict[str, Any]] = []
        with open(manifest) as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"Manifest line {line_number} must be a JSON object")
                missing = [
                    key for key in ("image", "prompt", "target") if key not in record
                ]
                if missing:
                    raise ValueError(
                        f"Manifest line {line_number} is missing fields: {missing}"
                    )
                if record.get("regime") is not None:
                    regime = int(record["regime"])
                    if regime not in {0, 1, 2}:
                        raise ValueError(
                            f"Manifest line {line_number} has invalid regime {regime}"
                        )
                else:
                    lon, lat = parse_action_pair(record["target"])
                    if not lon or not lat:
                        raise ValueError(
                            f"Manifest line {line_number} target must contain both "
                            "Longitudinal: and Lateral: actions"
                        )
                self.records.append(record)
                if max_samples is not None and len(self.records) >= max_samples:
                    break

        if not self.records:
            raise ValueError("Manifest contains no training samples")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        image_path = Path(rec["image"])
        if not image_path.is_absolute():
            image_path = self.root / image_path

        regime = rec.get("regime")
        if regime is None:
            regime = infer_regime(*parse_action_pair(rec["target"]))

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        return {
            "image": image,
            "prompt": rec["prompt"],
            "target": rec["target"],
            "regime": int(regime),
            "sample_id": rec.get("sample_id", str(idx)),
        }
