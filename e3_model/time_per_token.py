"""Experiment E3: predicted time per generated token on each machine measured in E2.

Model (from the paper plan): each weight is read once per token, so
    time per token = max(weight bytes / memory bandwidth, weights * measured time per weight).
The first term is the memory floor: what a machine with free decoding (native FP4 hardware) would reach
at the same memory bandwidth. The second term is the measured software decode-and-multiply cost.

Usage: python time_per_token.py
Reads ../results/e2_*.jsonl and ../results/v2/e1.jsonl; writes ../results/e3.jsonl and prints a table.
"""

import json
import pathlib
from dataclasses import dataclass

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
BUILD = "-O3 -ffast-math"
REGIME = "out_of_cache"
MEMORY_COPY = "memory_copy"
STREAMS_PER_COPY = 2  # a copy reads every byte and writes every byte
BYTES_PER_GIGABYTE = 1e9
NANOSECONDS_PER_SECOND = 1e9


@dataclass(frozen=True)
class Kernel:
    """One weight format as measured on one machine."""
    name: str
    bytes_per_weight: float
    seconds_per_weight: float


@dataclass(frozen=True)
class Machine:
    name: str
    bandwidth_bytes_per_second: float
    kernels: tuple


def main() -> None:
    machines = [machine_from(path) for path in sorted(RESULTS_DIR.glob("e2_*.jsonl"))]
    records = [prediction(machine, kernel, model, weights)
               for model, weights in quantized_weight_counts().items()
               for machine in machines
               for kernel in machine.kernels]
    with (RESULTS_DIR / "e3.jsonl").open("w") as results_file:
        results_file.writelines(json.dumps(record) + "\n" for record in records)
    print_table(records)


def machine_from(e2_results_path: pathlib.Path) -> Machine:
    rows = [json.loads(line) for line in e2_results_path.read_text().splitlines()]
    rows = [row for row in rows if row["build"] == BUILD]
    copy_rate = next(row["copy_gb_per_s"] for row in rows if row["format"] == MEMORY_COPY)
    kernels = tuple(Kernel(row["format"], row["bytes_per_weight"], row["ns_per_weight"] / NANOSECONDS_PER_SECOND)
                    for row in rows if row.get("regime") == REGIME)
    return Machine(rows[0]["machine"], copy_rate * STREAMS_PER_COPY * BYTES_PER_GIGABYTE, kernels)


def quantized_weight_counts() -> dict:
    """Linear-layer weights of each model, as counted by the E1 accuracy runs."""
    rows = [json.loads(line) for line in (RESULTS_DIR / "v2" / "e1.jsonl").read_text().splitlines()]
    return {row["model"]: row["quantized_weights"] for row in rows}


def memory_floor_seconds(weights: int, bytes_per_weight: float, bandwidth_bytes_per_second: float) -> float:
    return weights * bytes_per_weight / bandwidth_bytes_per_second


def decode_seconds(weights: int, seconds_per_weight: float) -> float:
    return weights * seconds_per_weight


def prediction(machine: Machine, kernel: Kernel, model: str, weights: int) -> dict:
    floor = memory_floor_seconds(weights, kernel.bytes_per_weight, machine.bandwidth_bytes_per_second)
    decode = decode_seconds(weights, kernel.seconds_per_weight)
    return {
        "model": model,
        "machine": machine.name,
        "format": kernel.name,
        "weights": weights,
        "megabytes": weights * kernel.bytes_per_weight / 1e6,
        "memory_floor_seconds": floor,
        "decode_seconds": decode,
        "seconds_per_token": max(floor, decode),
        "bound_by": "memory" if floor >= decode else "decode",
    }


def print_table(records: list) -> None:
    print("model | machine | format | MB | memory floor ms | measured decode ms | ms per token | bound by")
    for record in records:
        print(f"{record['model'].split('/')[-1]} | {record['machine']} | {record['format']} | "
              f"{record['megabytes']:.0f} | {1e3 * record['memory_floor_seconds']:.1f} | "
              f"{1e3 * record['decode_seconds']:.1f} | {1e3 * record['seconds_per_token']:.1f} | "
              f"{record['bound_by']}")


if __name__ == "__main__":
    main()
