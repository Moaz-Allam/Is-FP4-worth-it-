"""Zero-shot multiple-choice accuracy of one model under each weight format.

Usage: python run_tasks.py <experiment: a key of quant.EXPERIMENTS> <model_name> [format_name ...]
Appends one JSON line per format to ../results/<experiment>_tasks.jsonl.
"""

import json
import sys
import time

import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer

import quant
import run_accuracy

TASKS = ("arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande")
EXAMPLES_PER_TASK = 2000
BATCH_SIZE = 32


def main(experiment: str, model_name: str, format_names: tuple = ()) -> None:
    """Score the experiment's formats; when format names are given, only those."""
    device = run_accuracy.pick_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for weight_format in quant.EXPERIMENTS[experiment]:
        if format_names and weight_format.name not in format_names:
            continue
        started = time.time()
        model, _ = run_accuracy.load_quantized_model(model_name, weight_format, device)
        record = {
            **run_accuracy.format_description(model_name, weight_format),
            **task_accuracies(model, tokenizer),
            "examples_per_task": EXAMPLES_PER_TASK,
            "device": device,
            "seconds": round(time.time() - started, 1),
        }
        run_accuracy.append_record(record, run_accuracy.RESULTS_DIR / f"{experiment}_tasks.jsonl")
        print(json.dumps({key: value for key, value in record.items() if not isinstance(value, list)}),
              flush=True)
        del model
        run_accuracy.free_device_memory()


def task_accuracies(model, tokenizer) -> dict:
    harness_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=BATCH_SIZE)
    evaluation = lm_eval.simple_evaluate(model=harness_model, tasks=list(TASKS), limit=EXAMPLES_PER_TASK,
                                         log_samples=True)
    scores = evaluation["results"]
    accuracies = {task: scores[task]["acc,none"] for task in TASKS}
    standard_errors = {f"{task}_standard_error": scores[task]["acc_stderr,none"] for task in TASKS}
    correct_by_example = {f"{task}_correct": [int(sample["acc"]) for sample in evaluation["samples"][task]]
                          for task in TASKS}
    return {**accuracies, **standard_errors, **correct_by_example,
            "mean_accuracy": sum(accuracies.values()) / len(TASKS)}


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], tuple(sys.argv[3:]))
