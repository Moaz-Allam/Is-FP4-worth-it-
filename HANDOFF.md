# Handoff for Moaz — state of the experiments, 13 November 2026

This file is project notes, not paper text. Every sentence of the paper is still ours to write.
Numbers quoted from other papers (RaZeR, Kang, Chhugani, Fasoli) have **not** been re-checked in the PDFs yet; check them before citing.

## 1. What changed in the division of work

- Ziad ran Experiment 1 (accuracy), Experiment 4 (scale rules), a new Experiment 5 (scale sweep, see section 3), **and also Experiments 2 and 3** on a laptop and a server processor.
- What is left for you: **the SparkFun Edge board run of Experiment 2**. It is planned for the final submission (3 December), not for the draft. The draft uses the plan's fallback: laptop and server numbers, with the board as future work.
- Optional and useful: run Experiment 2 on your laptop too (one command, section 6). A third machine makes the timing table stronger.
- The professor's comment on the proposal asked for (a) the initial reference list, which is `references_verified.md` and must be in the draft, and (b) specialised hardware against 32-bit general-purpose hardware, which is what Experiments 2 and 3 answer.

## 2. What is in the repository

| Folder | Contents |
|---|---|
| `e1_accuracy/` | `quant.py` (all weight formats), `run_accuracy.py` (perplexity and drift), `run_tasks.py` (five task benchmarks), `test_quant.py` (20 tests) |
| `e2_cost/` | C kernels for a software-decoded dot product, their tests, the benchmark, a `Makefile` |
| `e3_model/` | `time_per_token.py`: predicted time per token from the Experiment 2 timings |
| `analysis/` | paired statistical tests, figure script, LaTeX table script |
| `results/v2/` | **the final accuracy results** (five models) and `paired_tests.jsonl` |
| `results/e2_*.jsonl`, `results/e3.jsonl` | timing results and the time-per-token predictions |
| `results/tables/` | six LaTeX tables (need the `booktabs` package) |
| `figures/` | three figures, each as PDF and PNG |
| `results/e1.jsonl`, `results/e4.jsonl`, `results/first_sweep_wikitext_only/` | earlier runs with the old protocol; kept for the record, **do not quote them** (the first sweep is the only place blocks of 8 and 64 appear: GPT-2 and Qwen, WikiText-2 only) |

The scripts that submitted the jobs to Kaggle are not in the repository (they are tied to Ziad's accounts). The code they ran is exactly `e1_accuracy/`.

## 3. Experiment 5: the scale sweep (the part that is ours)

**The question.** MXFP4 and NVFP4 both store 4-bit elements (E2M1) plus one scale per block. MXFP4 spends 8 scale bits on a power of two per 32 weights (4.25 bits per weight). NVFP4 spends 8 scale bits on a fractional scale (E4M3) per 16 weights (4.5 bits per weight). Ziad asked: why 8 bits? What happens with 6, 4, 2 or 0?

**The sweep.** E2M1 elements, blocks of 16 or 32, block scale stored as an unsigned float with E exponent bits and M mantissa bits, plus one full-precision scale per tensor. Storage is `4 + (E + M) / block` bits per weight. Splits tested: E0M0; E1M1, E2M0; E2M2, E3M1, E4M0; E3M3, E4M2; E4M4, E5M3. Names in the files look like `e2m1_b16_scale_e3m3`.

**Models.** GPT-2 (124M), TinyLlama-1.1B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-1.7B, Phi-3.5-mini (3.8B). Weights only, no calibration data, output head left unquantized.

**Measures.**
- Perplexity on WikiText-2 and on C4, with standard errors.
- **Drift**: the Kullback-Leibler divergence between the unquantized model's next-token predictions and the quantized model's, on 32 WikiText-2 windows. Lower is better. This is the measure that separates the formats; perplexity is noisier and task accuracy does not separate them at all.
- Five zero-shot tasks (ARC-Easy, ARC-Challenge, HellaSwag, PIQA, WinoGrande), first 2,000 questions each. GPT-2 has no task results on purpose (near chance).

**Drift results** (GPT-2 / DeepSeek / Qwen / TinyLlama / Phi-3.5):

| Format | Bits per weight | Drift |
|---|---|---|
| MXFP4 as published | 4.25 | 0.166 / 0.257 / 0.260 / 0.100 / 0.149 |
| MXFP4, best scale rule (threshold 7) | 4.25 | 0.143 / 0.257 / 0.212 / 0.073 / 0.129 |
| Block 32, 4-bit scale E2M2 | 4.125 | 0.143 / 0.196 / 0.148 / 0.061 / 0.113 |
| Block 32, 6-bit scale E3M3 | 4.1875 | 0.073 / 0.152 / 0.145 / 0.047 / 0.082 |
| Block 32, 8-bit scale E4M4 | 4.25 | 0.074 / 0.155 / 0.142 / 0.048 / 0.081 |
| Block 16, 4-bit scale E2M2 | 4.25 | 0.125 / 0.145 / 0.129 / 0.056 / 0.108 |
| Block 16, 6-bit scale E3M3 | 4.375 | 0.060 / 0.107 / 0.116 / 0.041 / 0.069 |
| Block 16, 8-bit scale E4M4 | 4.5 | 0.057 / 0.110 / 0.119 / 0.040 / 0.067 |
| NVFP4 as published | 4.5 | 0.061 / 0.128 / 0.131 / 0.044 / 0.074 |
| INT4, group of 32 | 4.5 | 0.091 / 0.185 / 0.222 / 0.059 / 0.120 |
| Block 16, 2-bit scale E2M0 | 4.125 | 0.418 / 0.197 / 0.207 / 0.092 / 0.209 |
| Block 16, 2-bit scale E1M1 | 4.125 | 4.495 / 0.482 / 0.596 / 0.382 / 1.532 |
| Block 16, no block scale E0M0 | 4.0 | 5.02 / 5.70 / 7.19 / 8.90 / 14.6 |

**What the paired tests support** (same windows for both formats; `results/v2/paired_tests.jsonl`):

1. A 6-bit scale and an 8-bit scale show no practical difference (gaps of 0.004 or less, in both directions).
2. A 4-bit scale is measurably worse than a 6-bit scale on all five models, but still usable.
3. At MXFP4's own storage (4.25 bits), both a block of 32 with a fractional 8-bit scale and a block of 16 with a 4-bit scale beat the best MXFP4 on all five models.
4. A block of 32 with a 4-bit scale (4.125 bits) beats published MXFP4 (4.25 bits) on all five models: less storage, less drift.
5. A block of 16 with a 6-bit scale (4.375 bits) beats NVFP4 (4.5 bits) on all five models; the margin is weakest on GPT-2.
6. With 2 scale bits the split decides everything: E2M0 works, E1M1 is badly damaged. With 0 bits every model collapses.
7. Task accuracy differs by 1 to 2 points between 4-bit formats, inside the noise.

**Wording to avoid.** "A 2-bit scale breaks the model" is wrong (E2M0 works). "Block 16 with a 4-bit scale matches NVFP4" holds only on Qwen and DeepSeek. Scales of 3 bits were never tested. Qwen's perplexity is erratic; trust drift for that model.

**What is new and what is not.**
- Not new: a 6-bit scale is as good as 8 bits (RaZeR, arXiv 2501.04052); a format near 4.25 bits can match NVFP4 (Kang et al., arXiv 2611.00486); 6-bit scales across block sizes (Fasoli et al., arXiv 2601.19026, Appendix H). On those points we are a replication on smaller models, and they must be cited.
- Appears new: one controlled grid that varies scale width and block size **together at equal storage** for 4-bit floating-point elements, including scales of 4 bits or fewer and the effect of the exponent/mantissa split.
- Limits to state: small models only (up to 3.8B weights), weights only (activations not quantized), no calibration, blocks of 16 and 32 only in the final protocol.

## 4. Experiment 4: MXFP4 scale rules

Eight variants in `results/v2/e4.jsonl` (`mxfp4_floor`, `mxfp4_up`, `mxfp4_best`, `mxfp4_threshold_7`, `mxfp4_threshold_7_25`, `mxfp4_three_quarters`, `nvfp4`, `nvfp4_four_over_six`). Threshold 7 is the best MXFP4 rule on drift and is the "best MXFP4" used in section 3. Table: `results/tables/scale_rules_drift.tex`.

## 5. Experiments 2 and 3: what software decode costs

**Experiment 2.** A matrix-times-vector product (4,096 activations per row) with the weights stored four ways: FP32, INT8 with one FP32 scale per 32 weights, MXFP4 decoded through a 16-entry table, and MXFP4 decoded two elements at a time through a 256-entry table. The plan said "dot product of length N"; a matrix with a fixed activation vector was used so that activation traffic does not hide the difference between weight formats. One core, plain C.

Nanoseconds per weight, `-O3 -ffast-math`, weights larger than the cache:

| Format | Bytes per weight | Apple M1 Pro | Intel Xeon 2.2 GHz |
|---|---|---|---|
| FP32 | 4 | 0.078 | 0.49 |
| INT8 + FP32 scale | 1.125 | 0.076 | 0.28 |
| MXFP4, 16-entry table | 0.531 | 0.227 | 1.55 |
| MXFP4, 256-entry pair table | 0.531 | 0.147 | 1.29 |

So on general-purpose hardware MXFP4 is 2 to 3 times **slower** per weight than FP32 even though it is 7.5 times smaller: the processor has no 4-bit circuits, so decoding dominates. Both builds (`-O3` and `-O3 -ffast-math`) are in the result files. With plain `-O3` the compiler does not vectorize the FP32 loop, which makes MXFP4 look faster than it should; report the fast-math build and say so.

**Experiment 3.** Time per token = the larger of (weight bytes ÷ memory bandwidth) and (weights × measured time per weight). The first term is the memory floor: what hardware that decodes 4-bit elements natively could reach. For Qwen3-1.7B, milliseconds per token (memory floor in brackets):

| Format | M1 Pro | Xeon |
|---|---|---|
| FP32 | 110 (64) | 689 (535) |
| INT8 | 106 (18) | 398 (150) |
| MXFP4, pair table | 206 (8.5) | 1,815 (71) |

Every case is limited by decoding, not by memory. The gap between MXFP4's measured time and its floor (about 24 times on the M1 Pro and 26 times on the Xeon) is the cost of not having 4-bit hardware. Figure: `figures/e3_roofline.pdf`.

**Limits to state.** One core; plain C, not hand-vectorized; default x86 instruction set on the Xeon; memory bandwidth estimated as twice the measured copy rate; no published Blackwell figures used yet; on the Xeon the pair table was faster in the plain build than in the fast-math build, which was not investigated; the timing covers the four formats above only, not the swept scale widths.

## 6. How to rerun things

Python 3.12 with torch 2.15, transformers 5.22, datasets 5.1, lm_eval 0.4.13, matplotlib, pytest.

```
# Experiment 2 on your machine (needs a C compiler and make); writes results/e2_<name>.jsonl
cd e2_cost && make test && make run MACHINE=moaz_laptop

# Experiment 3 from every results/e2_*.jsonl file; writes results/e3.jsonl
cd e3_model && python time_per_token.py

# Accuracy: experiment is one of e1, e4, e5_b16, e5_b32; needs a GPU for the larger models
cd e1_accuracy && python run_accuracy.py e5_b16 Qwen/Qwen3-1.7B 2048
cd e1_accuracy && python run_tasks.py e5_b16 Qwen/Qwen3-1.7B

# Paired tests, figures, tables (read results/v2 and results/e2_*, results/e3)
cd analysis && python paired_tests.py && python make_figures.py && python make_tables.py

# Tests
cd e1_accuracy && python -m pytest -q      # likewise in e3_model and analysis
```

Note that `run_accuracy.py` appends to `results/<experiment>.jsonl`, not to `results/v2/`; the v2 files were merged from the Kaggle outputs. After adding a machine to Experiment 2, rerun `time_per_token.py` and `make_figures.py`, then commit the new result file.

## 7. Open items

- Draft: write the prose; include the reference list; use the tables and figures above.
- Add to the reference list: RaZeR 2501.04052, Fasoli 2601.19026, Kang 2611.00486, Schaefer and Tabak 2605.08565, SpQR 2306.03078, VS-Quant 2102.04503, Chhugani 2603.08713.
- Verify every number quoted from another paper in its PDF.
- Final submission: SparkFun Edge run (Moaz); possibly 3-bit scales and timing of the swept scale widths (not decided).
