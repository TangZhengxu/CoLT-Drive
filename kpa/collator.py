"""Collator that turns decision samples into model-ready batches.

Loss is computed on the target tokens only: everything up to and including the
assistant generation prompt is masked out with -100.

``apply_chat_template(tokenize=True)`` is the single processing entry point,
which keeps vision token alignment consistent with inference.
"""

from __future__ import annotations

from typing import Any

import torch

# Matches the inference default in colt_drive.inference, so a model sees images
# at the same resolution during training and evaluation.
DEFAULT_MIN_PIXELS = 256 * 28 * 28
DEFAULT_MAX_PIXELS = 1280 * 28 * 28


class DrivingCollator:
    """Batch keyframe samples for a Qwen-VL style processor."""

    def __init__(
        self,
        processor,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        min_pixels: int = DEFAULT_MIN_PIXELS,
    ):
        self.processor = processor
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

    def _build_messages(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        image_block = {
            "type": "image",
            "image": sample["image"],
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }
        return [
            {
                "role": "user",
                "content": [image_block, {"type": "text", "text": sample["prompt"]}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["target"]}],
            },
        ]

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        all_inputs: list[dict[str, torch.Tensor]] = []
        prompt_lens: list[int] = []

        for sample in batch:
            messages = self._build_messages(sample)

            full_out = self.processor.apply_chat_template(
                messages, tokenize=True, return_dict=True,
                return_tensors="pt", add_generation_prompt=False,
            )
            # The Qwen3-VL processor can emit mm_token_type_ids inconsistent with
            # the grid metadata. The model infers modality from special tokens in
            # input_ids anyway, so drop it.
            full_out.pop("mm_token_type_ids", None)
            all_inputs.append(dict(full_out))

            # Re-render the user turn alone to find where the target begins.
            prompt_out = self.processor.apply_chat_template(
                [messages[0]], tokenize=True, return_dict=True,
                return_tensors="pt", add_generation_prompt=True,
            )
            prompt_lens.append(prompt_out["input_ids"].shape[1])

        inputs = self._pad_and_stack(all_inputs)

        labels = inputs["input_ids"].clone()
        for i, prompt_len in enumerate(prompt_lens):
            labels[i, :prompt_len] = -100
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        inputs["labels"] = labels

        if "regime" in batch[0]:
            inputs["regime"] = torch.tensor(
                [s["regime"] for s in batch], dtype=torch.long
            )

        return inputs

    def _pad_and_stack(
        self, all_inputs: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Right-pad single-sample processor outputs into one batch."""
        if len(all_inputs) == 1:
            return all_inputs[0]

        pad_token_id = self.processor.tokenizer.pad_token_id or 0
        max_len = max(inp["input_ids"].shape[1] for inp in all_inputs)

        batched: dict[str, Any] = {}
        for key in all_inputs[0]:
            tensors = [inp[key] for inp in all_inputs]

            if key == "input_ids":
                padded = torch.full(
                    (len(tensors), max_len), pad_token_id, dtype=tensors[0].dtype
                )
                for i, t in enumerate(tensors):
                    padded[i, : t.shape[1]] = t[0]
                batched[key] = padded

            elif key == "attention_mask":
                padded = torch.zeros((len(tensors), max_len), dtype=tensors[0].dtype)
                for i, t in enumerate(tensors):
                    padded[i, : t.shape[1]] = t[0]
                batched[key] = padded

            elif key in ("pixel_values", "image_grid_thw"):
                batched[key] = torch.cat(tensors, dim=0)

            else:
                try:
                    batched[key] = torch.cat(tensors, dim=0)
                except Exception:
                    batched[key] = tensors

        return batched
