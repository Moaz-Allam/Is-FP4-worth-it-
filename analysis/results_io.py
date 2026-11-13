"""Reads the experiment result files and names models and formats for tables and figures."""

import json
import pathlib

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
ACCURACY_EXPERIMENTS = ("e1", "e4", "e5_b16", "e5_b32")
MODEL_NAMES = {
    "openai-community/gpt2": "GPT-2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama-1.1B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "DeepSeek-R1-1.5B",
    "Qwen/Qwen3-1.7B": "Qwen3-1.7B",
    "microsoft/Phi-3.5-mini-instruct": "Phi-3.5-mini",
}
FORMAT_NAMES = {
    "fp16": "FP16", "int4_g128": "INT4, group 128", "int4_g32": "INT4, group 32",
    "mxfp4": "MXFP4", "nvfp4": "NVFP4", "mxfp4_floor": "MXFP4 round-down", "mxfp4_up": "MXFP4 round-up",
    "mxfp4_best": "MXFP4 lower-error", "mxfp4_threshold_7": "MXFP4 threshold 7",
    "mxfp4_threshold_7_25": "MXFP4 threshold 7.25", "mxfp4_three_quarters": "MXFP4 three-quarters",
    "nvfp4_four_over_six": "NVFP4 four-over-six",
    "fp32": "FP32", "int8_fp32_scale": "INT8, FP32 block scale",
    "mxfp4_pair_table": "MXFP4, two codes per lookup",
}


def read_lines(path: pathlib.Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines()]


def accuracy_records() -> dict:
    """(model, format) -> record with perplexity, drift and weight error."""
    return {(record["model"], record["format"]): record
            for experiment in ACCURACY_EXPERIMENTS
            for record in read_lines(RESULTS_DIR / "v2" / f"{experiment}.jsonl")}


def task_records() -> dict:
    """(model, format) -> record with task accuracies and per-question correctness."""
    return {(record["model"], record["format"]): record
            for experiment in ACCURACY_EXPERIMENTS
            for record in read_lines(RESULTS_DIR / "v2" / f"{experiment}_tasks.jsonl")}


def timing_records() -> list:
    return [record for path in sorted(RESULTS_DIR.glob("e2_*.jsonl")) for record in read_lines(path)]


def is_scale_sweep(format_name: str) -> bool:
    return format_name.startswith("e2m1_b")


def scale_sweep_parts(format_name: str) -> tuple:
    """'e2m1_b16_scale_e3m3' -> (block size 16, exponent bits 3, mantissa bits 3)."""
    _, block, _, scale = format_name.split("_")
    exponent_bits, mantissa_bits = scale[1:].split("m")
    return int(block[1:]), int(exponent_bits), int(mantissa_bits)


def format_label(format_name: str) -> str:
    if not is_scale_sweep(format_name):
        return FORMAT_NAMES[format_name]
    block_size, exponent_bits, mantissa_bits = scale_sweep_parts(format_name)
    return f"block {block_size}, scale E{exponent_bits}M{mantissa_bits}"
