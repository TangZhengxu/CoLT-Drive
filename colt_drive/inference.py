#!/usr/bin/env python3
"""Run a vision-language model over a CoLT-Drive split.

Writes the JSON schema that ``colt_drive.judge`` consumes, so the two stages can
run on different machines: inference needs GPUs, judging only needs an API key.

Supports torchrun data parallelism by sharding the manifest across ranks.

Usage:
    # Single GPU
    python -m colt_drive.inference \
        --model Qwen/Qwen3-VL-2B-Instruct \
        --data-dir data/vfull \
        --output results/qwen3vl_2b/vfull.json

    # 8 GPUs
    torchrun --nproc_per_node=8 -m colt_drive.inference \
        --model Qwen/Qwen3-VL-2B-Instruct \
        --data-dir data/vfull \
        --output results/qwen3vl_2b/vfull.json

    # With a trained KPA adapter
    python -m colt_drive.inference \
        --model /path/to/slerp_merged_backbone \
        --adapter /path/to/adapters/kpa_qwen3vl_2b/final_moe \
        --data-dir data/vfull \
        --output results/kpa/vfull.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .prompts import PROMPT_VARIANTS, SYSTEM_PROMPT

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a VLM over a CoLT-Drive split")
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--adapter", default=None,
                   help="Optional RegMoE adapter directory (moe_weights.pt + moe_config.json). "
                        "--model must be the same backbone the adapter was trained against; "
                        "the adapter is a correction relative to that backbone, not standalone")
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Split directory holding manifest.json and <sample_id>/ subdirs")
    p.add_argument("--output", type=Path, required=True, help="Output JSON path")
    p.add_argument("--prompt", choices=sorted(PROMPT_VARIANTS), default="complex",
                   help="'complex' is the structured decision interface (default); "
                        "'base' is a single open-ended question")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                   help="Skip samples already present in the per-rank progress files")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Write partial results and exit successfully when samples fail")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate data loading and output schema without loading weights")
    return p.parse_args()


def init_distributed(dry_run: bool = False) -> tuple[int, int, Any]:
    try:
        import torch
    except ModuleNotFoundError:
        if dry_run:
            return 0, 1, "cpu"
        raise

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.distributed.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


def load_manifest(split_dir: Path) -> list[str]:
    with (split_dir / "manifest.json").open() as f:
        manifest = json.load(f)
    return [str(x["sample_id"] if isinstance(x, dict) else x) for x in manifest]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def find_image(sample_dir: Path) -> Path:
    for ext in ("jpg", "jpeg", "png"):
        candidate = sample_dir / f"front_camera.{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No front_camera image in {sample_dir}")


def load_model(model_path: str, adapter_path: str | None, device):
    import torch
    from transformers import AutoProcessor

    from kpa.backbone import load_vlm

    model = load_vlm(model_path, torch_dtype=torch.bfloat16).to(device)
    processor = AutoProcessor.from_pretrained(
        model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )

    if adapter_path:
        from kpa.loading import try_load_moe_from_adapter_dir

        loaded = try_load_moe_from_adapter_dir(
            model, adapter_path, device, dtype=torch.bfloat16, verbose=True
        )
        if loaded is None:
            raise RuntimeError(f"No MoE adapter found in {adapter_path}")
        model = loaded

    model.eval()
    return model, processor


def build_messages(image_path: Path, context: str, prompt_template: str) -> list[dict]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{context.strip()}\n\n{prompt_template}"},
            ],
        },
    ]


def generate(model, processor, messages, max_new_tokens: int, device) -> str:
    import torch

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, repetition_penalty=1.1
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def load_progress(path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(f"WARNING: skipping truncated progress line in {path}", file=sys.stderr)
                    continue
                done[row["sample_id"]] = row
    return done


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists. Pass --overwrite to replace it.")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    sample_ids = load_manifest(args.data_dir)
    if args.max_samples is not None:
        sample_ids = sample_ids[: args.max_samples]

    rank, world_size, device = init_distributed(dry_run=args.dry_run)
    is_main = rank == 0
    work_dir = args.output.parent
    run_config = {
        "model": args.model,
        "adapter": args.adapter or "none",
        "data_dir": str(args.data_dir.resolve()),
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "dry_run": args.dry_run,
        "manifest_hash": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
    }
    run_id = hashlib.sha256(
        json.dumps(run_config, sort_keys=True).encode()
    ).hexdigest()[:12]
    progress_path = work_dir / f".{args.output.stem}_{run_id}_rank{rank}_progress.jsonl"
    if not args.resume:
        progress_path.unlink(missing_ok=True)

    cached = load_progress(progress_path) if args.resume else {}
    if args.resume and progress_path.exists():
        # Rewrite only valid, deduplicated rows so an interrupted final line
        # cannot swallow the next append.
        with progress_path.open("w") as f:
            for row in cached.values():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    done = {
        sid: row
        for sid, row in cached.items()
        if not str(row.get("response", "")).startswith("ERROR:")
    }
    my_ids = [sid for sid in sample_ids[rank::world_size] if sid not in done]

    if is_main:
        print(f"Model:  {args.model}" + (f" + adapter {args.adapter}" if args.adapter else ""))
        print(f"Split:  {args.data_dir}  ({len(sample_ids)} samples, {world_size} rank(s))")
        print(f"Prompt: {args.prompt}")
        print(f"Run ID: {run_id}")
        if done:
            print(f"Resume: {len(done)} already done on this rank")

    model = processor = None
    if not args.dry_run:
        model, processor = load_model(args.model, args.adapter, device)

    prompt_template = PROMPT_VARIANTS[args.prompt]
    errors = 0
    t_start = time.time()

    for i, sample_id in enumerate(my_ids, 1):
        sample_dir = args.data_dir / sample_id
        t0 = time.time()
        try:
            meta = read_json(sample_dir / "meta.json")
            context = (sample_dir / "prompt.txt").read_text().strip()

            if args.dry_run:
                response = f"DRY_RUN {sample_id}"
            else:
                messages = build_messages(find_image(sample_dir), context, prompt_template)
                response = generate(model, processor, messages, args.max_new_tokens, device)
        except Exception as exc:
            errors += 1
            meta = {}
            response = f"ERROR: {type(exc).__name__}: {exc}"

        # Deliberately no ground truth here. The judge reads gt.json from the
        # data directory, so predictions never carry a copy of the labels that
        # could go stale and silently skew scores.
        row = {
            "sample_id": sample_id,
            "obstacle_type": meta.get("obstacle_type", ""),
            "obstacle_category": meta.get("obstacle_category", ""),
            "position": meta.get("position", ""),
            "response": response,
            "inference_time_s": round(time.time() - t0, 2),
        }
        with progress_path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if is_main and (i % 10 == 0 or i == len(my_ids)):
            print(f"[rank0 {i}/{len(my_ids)}] {sample_id} "
                  f"({row['inference_time_s']}s, errors={errors})", flush=True)

    if world_size > 1:
        import torch

        torch.distributed.barrier()

    final_error_count = 0
    final_missing_count = 0
    if is_main:
        merged: dict[str, dict[str, Any]] = {}
        current_ids = set(sample_ids)
        for r in range(world_size):
            for row in load_progress(
                work_dir / f".{args.output.stem}_{run_id}_rank{r}_progress.jsonl"
            ).values():
                if row["sample_id"] in current_ids:
                    merged[row["sample_id"]] = row

        order = {sid: i for i, sid in enumerate(sample_ids)}
        ordered = sorted(merged.values(), key=lambda r: order[r["sample_id"]])
        final_error_count = sum(
            str(row.get("response", "")).startswith("ERROR:") for row in ordered
        )
        final_missing_count = len(sample_ids) - len(ordered)

        payload = {
            "summary": {
                "model": args.model,
                "adapter": args.adapter or "none",
                "data_dir": str(args.data_dir),
                "prompt": args.prompt,
                "max_new_tokens": args.max_new_tokens,
                "total_samples": len(ordered),
                "expected_samples": len(sample_ids),
                "error_count": final_error_count,
                "missing_count": final_missing_count,
                "num_ranks": world_size,
                "dry_run": args.dry_run,
                "total_time_s": round(time.time() - t_start, 1),
            },
            "samples": ordered,
        }
        with args.output.open("w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(ordered)} samples to {args.output}")

        if not final_error_count and not final_missing_count:
            for r in range(world_size):
                (
                    work_dir
                    / f".{args.output.stem}_{run_id}_rank{r}_progress.jsonl"
                ).unlink(missing_ok=True)
        else:
            print("Preserving progress files so failed samples can be retried.")

    if world_size > 1:
        import torch

        torch.distributed.destroy_process_group()

    if is_main and (final_error_count or final_missing_count) and not args.continue_on_error:
        raise RuntimeError(
            "Inference completed with "
            f"{final_error_count} failed and {final_missing_count} missing samples. "
            "Inspect the output or pass --continue-on-error to accept partial results."
        )


if __name__ == "__main__":
    main()
