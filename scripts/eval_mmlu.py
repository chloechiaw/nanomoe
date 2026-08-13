"""
MMLU evaluation for a *base* (pretrained, non-SFT) model.

nanochat's CORE bundle has no MMLU, and `tasks/mmlu.py` is wired into `chat_eval.py`, which
expects a chat model. But MMLU is the one benchmark every open MoE reports (OLMoE, Qwen1.5-MoE,
DeepSeek-V2-Lite), so a base-model number is needed to sit on the same axis as theirs.

This follows the standard Hendrycks 5-shot protocol, which is what those published numbers use:

  - few-shot exemplars come from the **dev** split and are **subject-matched** (MMLU's dev split
    holds exactly 5 per subject, which is where "5-shot" comes from),
  - the prompt is prefixed with "The following are multiple choice questions (with answers)
    about {subject}.",
  - scoring is over the answer **letters** A/B/C/D, not the answer text.

Rather than reimplement scoring, the whole few-shot prompt is baked into each item's `query` and
handed to nanochat's own `evaluate_task` with num_fewshot=0. That reuses the tested
multiple-choice scorer. Using `evaluate_task`'s built-in few-shot sampling would draw exemplars
at random from the test pool, mixing subjects, and would not be the standard protocol.

    python -m scripts.eval_mmlu --model-tag moe-d10-v3
    python -m scripts.eval_mmlu --model-tag moe-d10-v3 --max-examples 500   # quick check
"""

import argparse
import time
from collections import defaultdict

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, print0
from nanochat.core_eval import evaluate_example
from tasks.common import load_hub_dataset

LETTERS = ("A", "B", "C", "D")
HEADER = "The following are multiple choice questions (with answers) about {subject}."


def render_question(row):
    """One MMLU item, without the answer. Matches the Hendrycks format."""
    lines = [row["question"].strip()]
    for letter, choice in zip(LETTERS, row["choices"]):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def build_items(test_ds, dev_by_subject, num_fewshot):
    """Pre-render each test question into a full few-shot prompt in CORE's MC item format."""
    items = []
    for i in range(len(test_ds)):
        row = test_ds[i]
        subject = row["subject"]
        parts = [HEADER.format(subject=subject.replace("_", " "))]
        for shot in dev_by_subject.get(subject, [])[:num_fewshot]:
            parts.append(f"{render_question(shot)} {LETTERS[shot['answer']]}")
        parts.append(render_question(row))
        items.append({
            "query": "\n\n".join(parts),
            "choices": list(LETTERS),      # letter scoring, the standard MMLU convention
            "gold": row["answer"],
            "subject": subject,
        })
    return items


def main():
    p = argparse.ArgumentParser(description="MMLU eval for a base model")
    p.add_argument("--model-tag", type=str, default=None)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--num-fewshot", type=int, default=5)
    p.add_argument("--max-examples", type=int, default=-1, help="-1 = full 14,042-question test set")
    args = p.parse_args()

    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)
    model, tokenizer, _ = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)

    print0("Loading cais/mmlu ...")
    test_ds = load_hub_dataset("cais/mmlu", "all", split="test")
    dev_ds = load_hub_dataset("cais/mmlu", "all", split="dev")
    dev_by_subject = defaultdict(list)
    for i in range(len(dev_ds)):
        row = dev_ds[i]
        dev_by_subject[row["subject"]].append(row)

    items = build_items(test_ds, dev_by_subject, args.num_fewshot)
    if args.max_examples > 0:
        # stride rather than truncate, so all 57 subjects stay represented
        stride = max(1, len(items) // args.max_examples)
        items = items[::stride][:args.max_examples]
    print0(f"MMLU: {len(items):,} questions, {args.num_fewshot}-shot, {len(dev_by_subject)} subjects")

    task_meta = {"task_type": "multiple_choice", "num_fewshot": 0, "continuation_delimiter": " "}
    # Loop evaluate_example directly rather than calling evaluate_task: identical work on a
    # single GPU (evaluate_task is that loop plus an all_reduce), but it keeps the per-item
    # results so we can break the score down by subject.
    t0 = time.time()
    per_subject = defaultdict(lambda: [0, 0])  # subject -> [correct, total]
    n_correct = 0
    for idx, item in enumerate(items):
        ok = bool(evaluate_example(idx, model, tokenizer, items, device, task_meta))
        n_correct += ok
        per_subject[item["subject"]][0] += ok
        per_subject[item["subject"]][1] += 1
        if idx and idx % 2000 == 0:
            print0(f"  {idx:,}/{len(items):,}  running acc {n_correct / idx:.4f}")
    elapsed = time.time() - t0

    accuracy = n_correct / len(items)                                     # micro (per question)
    macro = sum(c / t for c, t in per_subject.values()) / len(per_subject)  # macro (per subject)

    ranked = sorted(per_subject.items(), key=lambda kv: -kv[1][0] / kv[1][1])
    print0("\nbest / worst subjects:")
    for name, (c, t) in ranked[:5] + ranked[-5:]:
        print0(f"  {name:44s} {c / t:6.3f}  ({c}/{t})")

    print0("=" * 62)
    print0("MMLU (5-shot, subject-matched dev exemplars, letter-scored)")
    print0(f"  accuracy (micro) : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print0(f"  accuracy (macro) : {macro:.4f}  ({macro * 100:.2f}%)")
    print0(f"  chance           : 0.2500  (4-way)")
    print0(f"  centered         : {(accuracy - 0.25) / 0.75:.4f}")
    print0(f"  questions        : {len(items):,}   subjects: {len(per_subject)}   time: {elapsed:.1f}s")
    print0("=" * 62)
    compute_cleanup()


if __name__ == "__main__":
    main()
