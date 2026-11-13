"""Paired comparisons between weight formats on the same text windows and the same task questions.

For formats A and B, every measure is the mean of (A - B) over the shared items, with its standard error
and z = mean / standard error. Negative loss and drift differences mean A is better; positive accuracy
differences mean A is better.

Usage: python paired_tests.py     (writes ../results/v2/paired_tests.jsonl and prints a table)
"""

import json
import math

import results_io

TASKS = ("arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande")
COMPARISONS = (
    ("e2m1_b16_scale_e3m3", "e2m1_b16_scale_e4m4", "6-bit vs 8-bit scale, block 16"),
    ("e2m1_b32_scale_e3m3", "e2m1_b32_scale_e4m4", "6-bit vs 8-bit scale, block 32"),
    ("e2m1_b16_scale_e2m2", "e2m1_b16_scale_e3m3", "4-bit vs 6-bit scale, block 16"),
    ("e2m1_b16_scale_e2m0", "e2m1_b16_scale_e2m2", "2-bit (E2M0) vs 4-bit scale, block 16"),
    ("e2m1_b16_scale_e1m1", "e2m1_b16_scale_e2m0", "2-bit E1M1 vs 2-bit E2M0, block 16"),
    ("e2m1_b32_scale_e4m4", "mxfp4_threshold_7", "block 32 E4M4 vs best MXFP4, both 4.25 bits"),
    ("e2m1_b16_scale_e2m2", "mxfp4_threshold_7", "block 16 E2M2 vs best MXFP4, both 4.25 bits"),
    ("e2m1_b16_scale_e2m2", "e2m1_b32_scale_e4m4", "block 16 E2M2 vs block 32 E4M4, both 4.25 bits"),
    ("e2m1_b32_scale_e2m2", "mxfp4", "block 32 E2M2 (4.125 bits) vs MXFP4 (4.25 bits)"),
    ("e2m1_b16_scale_e3m3", "nvfp4", "block 16 E3M3 (4.375 bits) vs NVFP4 (4.5 bits)"),
    ("nvfp4", "mxfp4", "NVFP4 vs MXFP4"),
)


def main() -> None:
    accuracy, tasks = results_io.accuracy_records(), results_io.task_records()
    rows = [comparison_row(model, first, second, description, accuracy, tasks)
            for first, second, description in COMPARISONS
            for model in results_io.MODEL_NAMES]
    with (results_io.RESULTS_DIR / "v2" / "paired_tests.jsonl").open("w") as output:
        output.writelines(json.dumps(row) + "\n" for row in rows)
    print_table(rows)


def paired_difference(first: list, second: list) -> dict:
    """Mean of first - second over paired items, its standard error, and their ratio z."""
    differences = [a - b for a, b in zip(first, second, strict=True)]
    mean = sum(differences) / len(differences)
    variance = sum((difference - mean) ** 2 for difference in differences) / (len(differences) - 1)
    standard_error = math.sqrt(variance / len(differences))
    return {"mean": mean, "standard_error": standard_error, "z": mean / standard_error if standard_error else 0.0}


def pooled_correctness(task_record: dict) -> list:
    return [float(answer) for task in TASKS for answer in task_record[f"{task}_correct"]]


def comparison_row(model, first, second, description, accuracy, tasks) -> dict:
    row = {"model": results_io.MODEL_NAMES[model], "comparison": description, "first": first, "second": second}
    one, two = accuracy[(model, first)], accuracy[(model, second)]
    row["loss"] = paired_difference(one["window_losses"], two["window_losses"])
    row["drift"] = paired_difference(one["window_kl_divergences"], two["window_kl_divergences"])
    if (model, first) in tasks and (model, second) in tasks:
        row["accuracy"] = paired_difference(pooled_correctness(tasks[(model, first)]),
                                            pooled_correctness(tasks[(model, second)]))
    return row


def cell(difference: dict) -> str:
    return f"{difference['mean']:+.4f} (z {difference['z']:+.1f})"


def print_table(rows: list) -> None:
    print("comparison (first - second) | model | loss per token | drift | task accuracy")
    for row in rows:
        accuracy = cell(row["accuracy"]) if "accuracy" in row else "-"
        print(f"{row['comparison']} | {row['model']} | {cell(row['loss'])} | {cell(row['drift'])} | {accuracy}")


if __name__ == "__main__":
    main()
