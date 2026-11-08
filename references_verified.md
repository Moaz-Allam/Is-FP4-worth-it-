# Reference sheet: "When Is FP4 Worth It?" (working notes, not paper text)

These are research notes, made with AI tools (SciSpace and alphaXiv search, Claude). The course rules say the paper text must be your own and every citation must be checked. So: open each PDF, confirm each number on the page given, and write the paper in your own words. Do not paste from this file.

Status key: **[read]** = full text read through alphaXiv, numbers below are from the PDF pages given. **[exists]** = confirmed to exist (title, authors, venue) through SciSpace, not read in full.

## A. Block formats and MX (foundations; 1-2 are course readings)

1. **[exists]** Drumond, Lin, Jaggi, Falsafi. *Training DNNs with Hybrid Block Floating Point.* NeurIPS 2018. arXiv:1804.01526. (course reading)
2. **[exists]** Darvish Rouhani et al. *Pushing the Limits of Narrow Precision Inferencing at Cloud Scale with Microsoft Floating Point.* NeurIPS 2020, 33:10271-10281. (syllabus)
3. **[read]** Darvish Rouhani et al. *Microscaling Data Formats for Deep Learning.* arXiv:2310.10537, 2023. (course reading)
   - p.3, Table 1: all MX formats use block size 32 and an E8M0 shared scale; MXFP4 elements are FP4 E2M1.
   - p.3, Algorithm 1: shared_exp = floor(log2(max|V|)) − emax_elem; X = 2^shared_exp; elements clamped. (This is HW3's X = 2^floor(log2 max)/4 rule, with emax = 2 for E2M1.)
   - p.6, Table 6, LLaMA-7B direct-cast WikiText perplexity: FP32 9.488; MXFP4 weights + MXFP8 activations 11.476; MXFP4 weights and activations 27.201.
4. **[exists]** Darvish Rouhani et al. *With Shared Microexponents, A Little Shifting Goes a Long Way.* ISCA 2023. (course reading)
5. **[exists]** Micikevicius et al. *FP8 Formats for Deep Learning.* arXiv:2209.05433, 2022.
6. **[exists]** OCP *Microscaling Formats (MX) Specification v1.0*, 2023. A standard: cite it, but it does not count as one of the 6 papers.

## B. 4-bit LLM quantization methods (context for the survey)

7. **[exists]** Frantar, Ashkboos, Hoefler, Alistarh. *GPTQ.* ICLR 2023. arXiv:2210.17323.
8. **[exists]** Lin et al. *AWQ.* MLSys 2024. arXiv:2306.00978.
9. **[exists]** Xiao et al. *SmoothQuant.* ICML 2023. arXiv:2211.10438.
10. **[exists]** Dettmers et al. *QLoRA* (NF4). NeurIPS 2023.
11. **[exists]** Ashkboos et al. *QuaRot.* NeurIPS 2024. arXiv:2404.00456.

## C. FP4 (MXFP4 / NVFP4) for LLMs

12. **[read]** Egiazarian, Panferov, Kuznedelev, …, Hoefler, Castro, Kurtić, Alistarh. *Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization.* **ICLR 2026.** arXiv:2509.23202.
    - p.2: bits per element are 4.5 for NVFP4 and 4.25 for MXFP4. NVIDIA Blackwell supports both; AMD will support MXFP4.
    - p.1-2: two problems. (1) NVFP4's small group size cancels traditional outlier mitigation. (2) MXFP4's power-of-two scale causes high error, about a 10% relative accuracy drop.
    - p.9, Table 1, Llama-3.1-8B-Instruct, W4A4, average recovery versus FP16 with round-to-nearest: INT4 92.63%, NVFP4 94.67%, MXFP4 87.83%. With their MR-GPTQ method, MXFP4 reaches 93.31%.
    - p.1: "FP4 is not an automatic upgrade over INT4."
    - p.10: on B200, layer speedup is about 3.6× out of an ideal 4×; end-to-end up to 2.2× over BF16 (RTX 5090: about 6× layer-wise, 4× end-to-end).
    - p.27, Appendix H: fitting MXFP's E8M0 scales to the data range raises Llama-3-8B round-to-nearest recovery from 87.8% to 94.3%. **This is prior work on MXFP4 scale selection.**
13. **[read]** Cook, Guo, Xiao, …, Han. *Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling.* arXiv:2512.02010 (preprint, under review).
    - p.2, p.5: quantize each block with its max scaled to 6 and to 4, keep the one with lower MSE (MSE rule best, Table 4). It uses the existing scale bits.
    - p.6, Table 3, WikiText-2 word perplexity, Llama-3 1B, W4A4 round-to-nearest: BF16 11.98, MXFP4 17.67, NVFP4 14.27.
    - p.9, §5.2: **does not work for MXFP4**, because the E8M0 scale can only change by factors of 2, and scaling to 4 needs a scale 1.5× larger.
14. **[read]** NVIDIA. *Pretraining Large Language Models with NVFP4.* arXiv:2509.25149, 2025.
    - p.2: NVFP4 uses 16-element blocks, an E4M3 block scale and an FP32 per-tensor scale.
    - p.3, Table 1, Blackwell Tensor Cores, speed versus BF16: MXFP4 and NVFP4 4× on GB200 and 6× on GB300. FP4 math is 2× (GB200) and 3× (GB300) faster than FP8.
    - p.9: MXFP4 needs 36% more training tokens than NVFP4 to reach the same loss on an 8B model.
    - p.17, Appendix B.4: MXFP4 scales are commonly rounded **up** to avoid saturation, which can waste the ±4 and ±6 values and up to one binade (2.58 instead of 3.58 binades used).
15. **[exists]** Tseng, Yu, Park. *Training LLMs with MXFP4.* arXiv:2502.20586, 2025. (cited by 13 and 14)
16. **[exists]** Chen et al. *INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats.* arXiv:2510.25602, 2025.
17. **[exists]** Fasoli et al. *Is Finer Better? The Limits of Microscaling Formats in Large Language Models.* 2026. Smaller blocks can make results worse.
18. **[exists]** Gupta et al. *Search Your Block Floating Point Scales!* 2026. Searched scales instead of max-based ones.
19. **[exists]** Chhugani et al. *Unveiling the Potential of Quantization with MXFP4: Strategies for Quantization Error Reduction* (Overflow-Aware Scaling). 2026.

## D. Specialized vs. 32-bit general-purpose hardware (for the professor's comment)

20. **[exists]** Khan. *De-Quantization Penalties for Interactive LLM Inference on Prosumer GPUs.* IEEE Computer Architecture Letters, 2026. On an RTX 3090 without native INT4 support, 4-bit models were 1.3-2.2× slower and 2.4× less energy-efficient than FP16 (from the abstract; read the paper).
21. **[exists]** Kim et al. *SqueezeLLM: Dense-and-Sparse Quantization.* 2023. Batch-1 generation is limited by memory bandwidth.
22. **[exists]** Guo et al. *Fast Matrix Multiplications for Lookup Table-Quantized LLMs* (FLUTE). 2024.
23. **[exists]** İslamoğlu et al. *MXDOTP: A RISC-V ISA Extension for Enabling Microscaling (MX) Floating-Point Dot Products.* 2025.
24. **[exists]** Wipfli et al. *VMXDOTP: A RISC-V Vector ISA Extension for Efficient Microscaling (MX) Format Acceleration.* 2026. Up to 7.0× over software-emulated MXFP8 matmul (abstract).
25. **[exists]** Garofalo et al. *XpulpNN: Accelerating Quantized Neural Networks on RISC-V Processors through ISA Extensions.* DATE 2020. Without sub-byte instructions, MCUs spend instructions packing and unpacking.

## What this changes in the plan

- **Novelty:** the "alternative block-scale selection" idea already exists (12 App. H, 13, 18, 19). Use the experiment to *test* a claim from them instead.
- **A good, citable E4 for MXFP4 at equal storage:** paper 14 (App. B.4) says rounding MXFP4 scales up avoids saturation but wastes range, while the OCP rule (paper 3, Algorithm 1) rounds down and clamps. Paper 13 notes the 4/6 trick cannot work with E8M0. So compare, per block, (a) the OCP floor rule, (b) round-up, and (c) the lower-MSE choice of X or 2X. All three use the same 8-bit scale, so storage is identical. Measure weight MSE and perplexity.
- **Numbers to compare your E1 results against:** paper 13's Table 3 (MXFP4 vs NVFP4 vs BF16 perplexity) and paper 12's Table 1 (INT4 vs NVFP4 vs MXFP4 recovery).
- **Hardware argument:** use papers 14 and 12 for native-FP4 speedups (4-6× math vs BF16, 2.2× end-to-end), papers 20, 24 and 25 for what happens without native support, and your own Cortex-M4 measurement (E2) as the 32-bit general-purpose case.
