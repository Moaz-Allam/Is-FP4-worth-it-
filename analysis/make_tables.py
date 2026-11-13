"""Writes LaTeX tables of the results into ../results/tables.

Usage: python make_tables.py
"""

import results_io

TABLES_DIR = results_io.RESULTS_DIR / "tables"
FORMAT_ROWS = ("fp16", "int4_g128", "int4_g32", "mxfp4", "nvfp4")
SCALE_RULE_ROWS = ("mxfp4_floor", "mxfp4_up", "mxfp4_best", "mxfp4_threshold_7", "mxfp4_threshold_7_25",
                   "mxfp4_three_quarters", "nvfp4", "nvfp4_four_over_six")
COLLAPSE_DRIFT = 10.0
TIMING_BUILD = "-O3 -ffast-math"
MODELS = tuple(results_io.MODEL_NAMES)


def main() -> None:
    TABLES_DIR.mkdir(exist_ok=True)
    records = results_io.accuracy_records()
    sweep_rows = sorted((name for model, name in records if model == MODELS[0] and results_io.is_scale_sweep(name)),
                        key=lambda name: (results_io.scale_sweep_parts(name)[0], records[(MODELS[0], name)]["bits_per_weight"]))
    write_table("formats_perplexity", accuracy_table(records, FORMAT_ROWS, "perplexity", "{:.2f}"))
    write_table("formats_drift", accuracy_table(records, FORMAT_ROWS, "kl_divergence", "{:.3f}"))
    write_table("scale_rules_drift", accuracy_table(records, SCALE_RULE_ROWS, "kl_divergence", "{:.3f}"))
    write_table("scale_sweep_drift", accuracy_table(records, sweep_rows, "kl_divergence", "{:.3f}"))
    write_table("scale_sweep_perplexity", accuracy_table(records, sweep_rows, "perplexity", "{:.2f}"))
    write_table("e2_time_per_weight", timing_table(results_io.timing_records()))


def accuracy_table(records: dict, format_names, measure: str, number_format: str) -> list:
    header = ["Format", "Bits/weight"] + [results_io.MODEL_NAMES[model] for model in MODELS]
    rows = [[results_io.format_label(name), f"{records[(MODELS[0], name)]['bits_per_weight']:g}"]
            + [number(records[(model, name)][measure], number_format) for model in MODELS]
            for name in format_names]
    return [header] + rows


def number(value: float, number_format: str) -> str:
    return number_format.format(value) if value < 1000 else f"{value:.2e}"


def timing_table(timings: list) -> list:
    machines = sorted({record["machine"] for record in timings})
    rows_by_format = {}
    for record in timings:
        if record["build"] == TIMING_BUILD and record.get("regime") == "out_of_cache":
            row = rows_by_format.setdefault(record["format"], [results_io.format_label(record["format"]),
                                                               f"{record['bytes_per_weight']:.3f}"] + [""] * len(machines))
            row[2 + machines.index(record["machine"])] = f"{record['ns_per_weight']:.3f}"
    return [["Format", "Bytes/weight"] + [f"{machine} (ns/weight)" for machine in machines]] + list(rows_by_format.values())


def write_table(name: str, rows: list) -> None:
    column_spec = "l" + "r" * (len(rows[0]) - 1)
    lines = [f"\\begin{{tabular}}{{{column_spec}}}", "\\toprule", latex_row(rows[0]), "\\midrule"]
    lines += [latex_row(row) for row in rows[1:]] + ["\\bottomrule", "\\end{tabular}"]
    (TABLES_DIR / f"{name}.tex").write_text("\n".join(lines) + "\n")


def latex_row(cells: list) -> str:
    return " & ".join(cell.replace("_", "\\_") for cell in cells) + " \\\\"


if __name__ == "__main__":
    main()
