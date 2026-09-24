#!/usr/bin/env python3
"""Run MathTutorBench against a complete tutoring *system* and build a submission.

The model leaderboard evaluates a model. This script evaluates a system: a
product, an agentic workflow, a prompt chain, anything that answers behind an
OpenAI-compatible ``/v1/chat/completions`` endpoint. Internals stay private; the
endpoint is the only contract.

    python submit_system.py \
        --endpoint http://127.0.0.1:8777/v1 \
        --model my-system \
        --manifest systems/my-system.yaml

What it does beyond ``main.py``:

  * times every request, so wall-clock is measured by one clock under one
    protocol instead of being self-reported;
  * detects task-instruction substitution empirically, by comparing the two
    tasks that share a dataset;
  * refuses to emit a submission whose manifest is missing or incomplete.

See SYSTEMS.md for the rules.
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm

from models.completion_api import create_llm_model, LLMConfig
# Import order matters: tasks.base must land before registry, or tasks/__init__
# pulls registry back in half-built. main.py relies on the same ordering.
from tasks.base import TaskConfig
from registry import TaskRegistry

# Tasks that ask for an open-ended teacher turn, scored by the reward model.
# These are the four the systems leaderboard ranks on. The other five ask for a
# direct answer, which many tutoring systems are built never to give; a system
# may report them, but it is not required to.
PEDAGOGICAL_TASKS = [
    "scaffolding_generation",
    "pedagogy_following",
    "scaffolding_generation_hard",
    "pedagogy_following_hard",
]

# Task pairs that read the same dataset and differ only by the instruction given
# to the tutor. Verified against configs/: scaffolding_generation and
# pedagogy_following both load datasets/mathdial_bridge.json; the _hard pair both
# load datasets/mathdial_bridge_hard.json. A system that substitutes its own
# prompt therefore receives byte-identical input on both halves of a pair.
SHARED_DATASET_PAIRS = [
    ("scaffolding_generation", "pedagogy_following"),
    ("scaffolding_generation_hard", "pedagogy_following_hard"),
]

# Every field a submission must declare. The point is not bureaucracy: each one
# is something that moved a published number and could not be read off the score.
MANIFEST_FIELDS = [
    "system_name",
    "system_url",
    "contact",
    "models",                     # every model the system may call, by name
    "llm_calls_per_item_mean",
    "llm_calls_per_item_max",
    "task_instruction_honored",   # or replaced by the system's own prompt
    "reads_any_gold_field",       # reference_solution & co. MUST be false
    "external_tools",             # solvers, retrievers, calculators, search
    "samples_per_item",
    "temperature",
    "seed",
    "full_test_set",              # false if subsampled; say how many
    "ablation_no_tools",          # required, see SYSTEMS.md
    "notes",                      # caveats you would want attached to your row
]


def load_task_config(config_name):
    with open(Path("configs") / (config_name + ".yaml")) as f:
        return TaskConfig(**yaml.safe_load(f))


def percentile(values, q):
    """Nearest-rank percentile. No numpy dependency for four numbers."""
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(q / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[k]


def run_task(model, task_name, limit=0):
    """Run one task. Returns (metrics, generations, latencies, predictions)."""
    task_config = load_task_config(task_name)
    task = TaskRegistry.get_task(task_config.name)(task_config)

    examples = task.get_test_examples()
    if limit:
        examples = examples[:limit]

    predictions, targets, generations, latencies = [], [], [], []
    failures = 0

    for example in tqdm(examples, desc=task_name):
        example["shots"] = task_config.few_shot_samples
        started = time.perf_counter()
        try:
            response = model.generate(
                messages=[],
                system_prompt=task.get_system_prompt(example),
                stop=task_config.stop,
            ) or ""
        except Exception as exc:                     # one bad item must not kill a run
            print("request failed: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
            response, failures = "", failures + 1
        # Timed around the whole call, retries included. That is what a student
        # would have waited for, which is the number we care about.
        latencies.append(time.perf_counter() - started)

        prediction = task.parse_response(response)
        predictions.append(prediction)
        targets.append(task.format_ground_truth(example))
        generations.append({
            "problem": example.get("question", ""),
            "reference_solution": example.get("reference_solution", "N/A"),
            "dialog_history": example.get("conversation_json", []),
            "dialog_formatted": example.get("dialog_history", ""),
            "ground_truth_response": example.get("ground_truth_response", ""),
            "generated_teacher_utterance": prediction,
        })

    metrics = task.compute_metrics(predictions, targets)
    metrics["failed_requests"] = failures
    return metrics, generations, latencies, predictions


def instruction_substitution_check(predictions_by_task):
    """Did the system read the task instruction, or replace it with its own?

    The two tasks in a pair differ only by that instruction. If a system returns
    byte-identical turns on both, it did not read it, and the pair is one
    experiment reported twice rather than two independent tasks.

    The test is deliberately one-sided. Identical output proves substitution.
    Different output proves nothing, since a system with sampling enabled varies
    anyway. So it can catch an inflated claim, never certify an honest one, and
    it is reported as such.
    """
    findings = []
    for left, right in SHARED_DATASET_PAIRS:
        if left not in predictions_by_task or right not in predictions_by_task:
            continue
        a, b = predictions_by_task[left], predictions_by_task[right]
        n = min(len(a), len(b))
        if not n:
            continue
        identical = sum(1 for i in range(n) if a[i].strip() == b[i].strip())
        share = identical / n
        findings.append({
            "pair": [left, right],
            "compared": n,
            "identical_turns": identical,
            "identical_share": round(share, 4),
            # A handful of collisions is normal on short turns; a majority is not.
            "verdict": "task_instruction_replaced" if share >= 0.5 else "inconclusive",
        })
    return findings


def read_manifest(path):
    if not path:
        return None, ["no --manifest given"]
    manifest = yaml.safe_load(Path(path).read_text()) or {}
    problems = [f"missing field: {f}" for f in MANIFEST_FIELDS if f not in manifest]
    if manifest.get("reads_any_gold_field") is True:
        problems.append(
            "reads_any_gold_field is true. reference_solution and the other gold "
            "fields are the scoring ground truth; a system that reads them is not "
            "measurable and cannot be submitted.")
    return manifest, problems


def score_with_reward_model(output_dir, model_tag, tasks):
    """Shell out to the existing reward-model scorer, one call per task.

    Kept as a subprocess on purpose: reward_model/compute_scaffolding_score.py
    writes back into ./results/results-<model>.yaml by convention, and pulling it
    into this process would mean rewriting a script the leaderboard already uses.
    """
    scores = {}
    for task in tasks:
        data_path = Path(output_dir) / f"generations-{model_tag}-{task}.json"
        if not data_path.exists():
            continue
        subprocess.run(
            [sys.executable, "reward_model/compute_scaffolding_score.py",
             "--data_path", str(data_path)],
            check=True)
        written = Path("scaffolding_scores") / f"{data_path.name}_results.json"
        if written.exists():
            scores[task] = json.loads(written.read_text())
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a complete tutoring system and build a leaderboard submission")
    parser.add_argument("--endpoint", required=True,
                        help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8777/v1")
    parser.add_argument("--model", default=None,
                        help="model id served by the endpoint (default: the first one)")
    parser.add_argument("--api-key", dest="api_key", default=None)
    parser.add_argument("--tasks", default=",".join(PEDAGOGICAL_TASKS),
                        help="comma-separated task names")
    parser.add_argument("--manifest", default=None,
                        help="YAML declaration, required for a real submission")
    parser.add_argument("--output", default="results",
                        help="output directory (keep 'results' if you use --score)")
    parser.add_argument("--limit", type=int, default=0,
                        help="first N examples per task, for a smoke run (0 = full test set)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=2048)
    parser.add_argument("--score", action="store_true",
                        help="also run the pedagogical reward model on the generations")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    manifest, manifest_problems = read_manifest(args.manifest)

    model = create_llm_model(LLMConfig(
        provider="completion_api",
        model=args.model,
        base_url=args.endpoint,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        is_chat=True,
    ))
    model_tag = model.config.model.split("/")[-1]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results, latencies_by_task, predictions_by_task = {}, {}, {}
    for task in tasks:
        metrics, generations, latencies, predictions = run_task(model, task, args.limit)
        results[task] = metrics
        latencies_by_task[task] = latencies
        predictions_by_task[task] = predictions
        (output_dir / f"generations-{model_tag}-{task}.json").write_text(
            json.dumps(generations, indent=2))

    # Reported next to the win rate, never ranked on. A system that searches for
    # ten minutes a turn can top any quality table and still be unusable with a
    # student who is stuck at half past nine at night.
    all_latencies = [x for xs in latencies_by_task.values() for x in xs]
    latency = {
        "median_s": round(statistics.median(all_latencies), 2) if all_latencies else None,
        "p90_s": round(percentile(all_latencies, 90), 2) if all_latencies else None,
        "max_s": round(max(all_latencies), 2) if all_latencies else None,
        "per_task_median_s": {t: round(statistics.median(v), 2)
                              for t, v in latencies_by_task.items() if v},
        "measured_by": "submit_system.py, around each request, retries included",
        "caveat": "wall-clock depends on the submitter's hardware and network; "
                  "read it as an order of magnitude, not a ranking",
    }

    submission = {
        "system": model_tag,
        "tasks": results,
        "latency": latency,
        "instruction_substitution_check": instruction_substitution_check(predictions_by_task),
        "run": {
            "full_test_set": args.limit == 0,
            "limit": args.limit or None,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "endpoint": args.endpoint,
        },
        "manifest": manifest,
        "manifest_problems": manifest_problems,
    }

    if args.score:
        if output_dir.name != "results":
            print("warning: --score expects --output results (the scorer writes "
                  "back into ./results by convention)", file=sys.stderr)
        submission["reward_model"] = score_with_reward_model(output_dir, model_tag, tasks)

    path = output_dir / f"submission-{model_tag}.json"
    path.write_text(json.dumps(submission, indent=2))
    print(json.dumps(submission, indent=2))

    if manifest_problems:
        print("\nThis run is NOT submittable yet:", file=sys.stderr)
        for problem in manifest_problems:
            print("  - " + problem, file=sys.stderr)
        print("Start from systems/manifest.template.yaml.", file=sys.stderr)
        sys.exit(1)

    print("\nWrote " + str(path))


if __name__ == "__main__":
    main()
