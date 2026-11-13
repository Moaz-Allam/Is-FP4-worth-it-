"""Draws the paper's three figures from the result files into ../figures (PDF and PNG).

Usage: python make_figures.py
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

import results_io

FIGURES_DIR = pathlib.Path(__file__).resolve().parent.parent / "figures"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
TIMING_BUILD = "-O3 -ffast-math"
TIMING_REGIME = "out_of_cache"
ROOFLINE_MODEL = "Qwen/Qwen3-1.7B"
MACHINE_NAMES = {"m1_pro": "Apple M1 Pro", "kaggle_x86": "Intel Xeon 2.2 GHz"}
STREAMS_PER_COPY = 2
BYTE_TICKS = (0.5, 1, 2, 4)
PAIR_TABLE_FORMAT = "mxfp4_pair_table"
PUBLISHED_POINTS = (  # format, legend label, colour, marker
    ("mxfp4", "MXFP4 (published)", AQUA, "^"),
    ("nvfp4", "NVFP4 (published)", YELLOW, "D"),
    ("int4_g32", "INT4, group 32", MAGENTA, "X"),
)
SWEEP_LINES = ((16, "Scale sweep, block 16", BLUE, "o"), (32, "Scale sweep, block 32", ORANGE, "s"))


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    save(drift_against_storage(results_io.accuracy_records()), "drift_vs_bits")
    save(time_per_weight(results_io.timing_records()), "e2_time_per_weight")
    save(roofline(results_io.timing_records(), results_io.accuracy_records()), "e3_roofline")


def use_plain_log_ticks(axis_side) -> None:
    """Label a log axis with ordinary numbers at 1, 2 and 5 times each power of ten."""
    axis_side.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    axis_side.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis_side.set_minor_formatter(NullFormatter())


def save(figure, name: str) -> None:
    for extension in ("pdf", "png"):
        figure.savefig(FIGURES_DIR / f"{name}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(figure)


def best_per_scale_width(records: dict, model: str, block_size: int) -> list:
    """(bits per weight, lowest drift) for each scale width of one block size, leaving out the 0-bit scale."""
    best = {}
    for (record_model, name), record in records.items():
        if record_model != model or not results_io.is_scale_sweep(name):
            continue
        block, exponent_bits, mantissa_bits = results_io.scale_sweep_parts(name)
        if block != block_size or exponent_bits + mantissa_bits == 0:
            continue
        bits = record["bits_per_weight"]
        best[bits] = min(best.get(bits, float("inf")), record["kl_divergence"])
    return sorted(best.items())


def drift_against_storage(records: dict):
    figure, axes = plt.subplots(1, len(results_io.MODEL_NAMES), figsize=(11, 2.6), sharex=True, constrained_layout=True)
    for axis, (model, model_name) in zip(axes, results_io.MODEL_NAMES.items()):
        for block_size, label, colour, marker in SWEEP_LINES:
            points = best_per_scale_width(records, model, block_size)
            axis.plot(*zip(*points), color=colour, marker=marker, markersize=5, linewidth=2, label=label)
        for name, label, colour, marker in PUBLISHED_POINTS:
            record = records[(model, name)]
            axis.scatter(record["bits_per_weight"], record["kl_divergence"], color=colour, marker=marker, s=45,
                         edgecolor=SURFACE, linewidth=1, zorder=3, label=label)
        axis.set_yscale("log")
        use_plain_log_ticks(axis.yaxis)
        axis.set_title(model_name, fontsize=9)
        axis.grid(True, which="both", color=GRID, linewidth=0.5)
        axis.set_xlabel("Bits per weight")
    axes[0].set_ylabel("Drift from FP16 (KL divergence, log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.14))
    return figure


def out_of_cache_timings(timings: list, machine: str) -> list:
    return [record for record in timings if record["machine"] == machine and record["build"] == TIMING_BUILD
            and record.get("regime") == TIMING_REGIME]


def time_per_weight(timings: list):
    figure, axes = plt.subplots(1, len(MACHINE_NAMES), figsize=(7, 2.4))
    for axis, (machine, machine_name) in zip(axes, MACHINE_NAMES.items()):
        rows = out_of_cache_timings(timings, machine)
        labels = [f"{results_io.format_label(row['format'])}\n{row['bytes_per_weight']:.3f} bytes" for row in rows]
        bars = axis.barh(labels, [row["ns_per_weight"] for row in rows], color=BLUE, height=0.6)
        axis.bar_label(bars, fmt="%.2f", padding=3, color=INK)
        axis.invert_yaxis()
        axis.set_title(machine_name, fontsize=9)
        axis.set_xlabel("Nanoseconds per weight")
        axis.margins(x=0.18)
        axis.grid(True, axis="x", color=GRID, linewidth=0.5)
        axis.set_axisbelow(True)
    figure.tight_layout()
    return figure


def bandwidth_bytes_per_second(timings: list, machine: str) -> float:
    copy_rate = next(record["copy_gb_per_s"] for record in timings if record["machine"] == machine
                     and record["build"] == TIMING_BUILD and record["format"] == "memory_copy")
    return copy_rate * STREAMS_PER_COPY * 1e9


def roofline(timings: list, records: dict):
    weights = records[(ROOFLINE_MODEL, "fp16")]["quantized_weights"]
    figure, axes = plt.subplots(1, len(MACHINE_NAMES), figsize=(7, 2.8), sharex=True, constrained_layout=True)
    for axis, (machine, machine_name) in zip(axes, MACHINE_NAMES.items()):
        bandwidth = bandwidth_bytes_per_second(timings, machine)
        byte_range = (0.4, 5.0)
        axis.plot(byte_range, [1e3 * weights * size / bandwidth for size in byte_range], color=MUTED, linewidth=2,
                  label="Memory floor (decode in hardware)")
        rows = out_of_cache_timings(timings, machine)
        axis.scatter([row["bytes_per_weight"] for row in rows],
                     [1e-6 * weights * row["ns_per_weight"] for row in rows],
                     color=ORANGE, s=40, edgecolor=SURFACE, linewidth=1, zorder=3, label="Measured software decode")
        for row in rows:
            is_pair_table = row["format"] == PAIR_TABLE_FORMAT
            axis.annotate("MXFP4, two per lookup" if is_pair_table else results_io.format_label(row["format"]).split(",")[0],
                          (row["bytes_per_weight"], 1e-6 * weights * row["ns_per_weight"]),
                          textcoords="offset points", xytext=(6, -9 if is_pair_table else 4), fontsize=7, color=INK)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(BYTE_TICKS, [f"{tick:g}" for tick in BYTE_TICKS])
        axis.xaxis.set_minor_formatter(NullFormatter())
        use_plain_log_ticks(axis.yaxis)
        axis.set_title(f"{machine_name}, {results_io.MODEL_NAMES[ROOFLINE_MODEL]}", fontsize=9)
        axis.set_xlabel("Bytes per weight (log scale)")
        axis.grid(True, which="both", color=GRID, linewidth=0.5)
    axes[0].set_ylabel("Milliseconds per token (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.12))
    return figure


if __name__ == "__main__":
    main()
