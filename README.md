# CoLT-Drive

A counterfactual long-tail benchmark for driving decisions, plus KPA, a
regime-aware adaptation recipe for driving vision-language models.

The driving context (road, ego pose, navigation command, motion history) is held
fixed and only the object in the scene changes. The benchmark asks one question:
does the model's decision update when it should, and hold when it should not?

- **Data**: [`tangzx2024/CoLT-Drive`](https://huggingface.co/datasets/tangzx2024/CoLT-Drive) on the Hugging Face Hub
- **Task**: given a front-camera image and the driving context, output one
  longitudinal action and one lateral action

| Axis | Actions |
|---|---|
| Longitudinal | `keep_speed`, `slow_down`, `yield`, `creep`, `stop`, `speed_up` |
| Lateral | `keep_lane`, `nudge_left`, `nudge_right` |

Each sample has an *accepted set* of action pairs; a prediction is correct if it
is in the set. Two splits of 1,768 samples share identical targets: `vfull` has
naturalistic background traffic, `vclean` has it removed. Samples cover 50
obstacle types in five categories: `entity_living`, `entity_nonliving`,
`road_hazard`, `full_block`, and `false_positive`.

## Install

Python 3.10 or newer.

```bash
git clone https://github.com/tangzhengxu/CoLT-Drive && cd CoLT-Drive
pip install -r requirements.txt
```

Scoring only needs `requests`; the rest is for running models locally.

## Evaluate a model

**1. Download the data**

```bash
hf download tangzx2024/CoLT-Drive --repo-type dataset --local-dir data
```

Each split is `data/<split>/manifest.json` plus one folder per sample holding
`front_camera.jpg`, `prompt.txt`, `gt.json`, and `meta.json`.

**2. Run inference**

```bash
python -m colt_drive.inference \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --data-dir data/vfull \
    --output results/qwen3vl_2b/vfull.json
```

Use `torchrun --nproc_per_node=N -m colt_drive.inference ...` for multiple GPUs,
`--resume` to continue an interrupted run, and `--prompt base` for the
unstructured prompt variant.

To score outputs from your own pipeline instead, write them as:

```json
{"samples": [{"sample_id": "NP01_O10_center_v_full", "response": "<full model output>"}]}
```

**3. Score**

```bash
export JUDGE_API_KEY=...   # any OpenAI-compatible chat-completions endpoint
python -m colt_drive.judge \
    --results-file results/qwen3vl_2b/vfull.json \
    --data-dir data/vfull \
    --output-dir results/qwen3vl_2b/vfull_judged
```

The judge extracts the final decision from the free-form response, maps it to
the canonical actions with an LLM, and checks membership in the accepted set.
Set `JUDGE_API_ENDPOINT` and `JUDGE_MODEL` to change the endpoint (default:
OpenAI, `gpt-4o`). Use the same judge model for every system you compare.
Results are written to `summary.json` and `per_sample.json`; re-running resumes
from `progress.jsonl`.

The prompts in `colt_drive/prompts.py` are fixed. Scores are only comparable
when the prompt is byte-identical, so add a new constant rather than editing one.

## KPA

KPA adapts a driving VLM in three steps.

**1. Structured decision interface.** The prompt in `colt_drive/prompts.py`
leads the model from perception to a parsable action pair. No training.

**2. SLERP initialization.** Take a small spherical step from the pretrained
weights toward a driving expert, then freeze the merged backbone.

```bash
python -m kpa.slerp_merge \
    --base Qwen/Qwen3-VL-2B-Instruct \
    --finetuned /path/to/driving_lora_adapter \
    --output checkpoints/merged \
    --alphas 0.08
```

**3. Regime-aware adapter (RegMoE).** Train a LoRA mixture of experts on the
frozen backbone. During training the router receives a bias per behaviour
regime (cruising, maneuvering, reacting), read off the supervised action pair;
at inference no regime label is needed.

```bash
torchrun --nproc_per_node=8 -m kpa.train --config kpa/configs/kpa_qwen3vl_2b.yaml
```

Training data is a JSONL manifest with `image`, `prompt`, and `target` fields
(see `kpa/data.py`). The adapter is written to `<output_dir>/final_moe/` and
loads with:

```bash
python -m colt_drive.inference \
    --model checkpoints/merged_alpha008 \
    --adapter adapters/kpa_qwen3vl_2b/final_moe \
    --data-dir data/vfull \
    --output results/kpa/vfull.json
```

Trained weights are not distributed: they derive from a driving corpus that is
not public. An adapter only works with the merged backbone it was trained on.

## Layout

```text
colt_drive/   prompts.py, inference.py, judge.py
kpa/          moe_layers.py, regimes.py, slerp_merge.py, train.py,
              data.py, collator.py, loading.py, backbone.py, configs/
```

## License

Apache License 2.0. See [LICENSE](LICENSE). The dataset is distributed
separately on the Hugging Face Hub; see its dataset card for license terms.
