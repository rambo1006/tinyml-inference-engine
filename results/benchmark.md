# Benchmark Results

## Model

3-layer MLP: `13 → 64 → 32 → 3` (MFCC features → keyword class)
Classes: `yes`, `no`, `stop`

---

## Headline result: quantization accuracy

| Stage              | Accuracy | Notes                                   |
|---------------------|----------|------------------------------------------|
| Float32 (PyTorch)   | 83.0%    | Best validation accuracy over 30 epochs   |
| Int8 (Python sim)   | 81.7%    | Quantized weights, simulated in numpy     |
| Int8 (C engine)     | 81.7%    | Same weights, real C inference, 300/300 validation samples checked against Python |
| **Accuracy drop**   | **1.3%** | Float32 → int8, full validation set       |

This is the core result of the project. The C engine was verified against the Python int8 simulation by exporting the real validation set (`data/X_val.npy`, `data/y_val.npy`) into a C header and running all 300 samples through the actual `infer()` function — not a synthetic test. Result: **245/300 correct, exactly matching Python's 81.7%.**

A 1.3% accuracy drop from per-layer asymmetric quantization (separate scale + zero-point for every weight tensor and every activation, computed from each tensor's own min/max) is a tight result — naive single-scale or symmetric-only quantization schemes typically lose several percentage points more. This is the part of the project that required real engineering: getting the zero-point math, int32 accumulation, and float-domain rescaling-with-bias correct enough that a from-scratch C implementation reproduces a PyTorch reference bit-for-bit on the final predictions.

---

## Honest ceiling: why 81.7% and not higher

81.7% on 3-class keyword classification is well above the 33% random baseline, but it's not a strong number in absolute terms, and the model is very likely not the bottleneck. The feature extraction step (`week3_dataset.py`) collapses each 1-second audio clip into 13 MFCC coefficients by averaging across the entire time axis (`mfcc.mean(axis=1)`). That throws away essentially all temporal structure in the audio — "yes" and "no" don't just differ in their average spectral content, they differ in how that content evolves over the clip, and a mean collapses that signal away before the model ever sees it.

The accuracy ceiling here is the feature representation, not the network. A frame-level feature sequence (even fed into the same MLP per-frame, or into a small 1D CNN/RNN) would very likely outperform this by a meaningful margin. The mean-MFCC approach was a deliberate simplification to keep the inference engine itself — the actual point of this project — small and easy to reason about in hand-written C.

---

## Memory footprint

| Metric                  | Value        |
|---------------------------|---------------|
| Float32 weights            | 11,904 bytes (11.6 KB) |
| Int8 weights                | 2,976 bytes (2.9 KB)   |
| Memory reduction            | 4.0x                    |
| Runtime memory (host build: weights + biases + activation buffers) | 3,877 bytes (3.79 KB) |

For context: this model is small enough (11.6 KB even un-quantized) that it would have fit inside a typical Cortex-M4's 256 KB SRAM budget without any quantization at all. **Memory pressure was not the reason to quantize here** — at this model size, quantization is not solving a constraint that actually existed. The reasons quantization was still worth implementing are (1) it's the technique that makes larger models viable on real microcontrollers, and a 13→64→32→3 MLP is a deliberately small, easy-to-verify proof of the method, and (2) int8 integer arithmetic is generally cheaper in latency and power than float arithmetic on Cortex-M class cores, including ones with an FPU — that's a real benefit independent of memory headroom. The memory savings here are a side effect of implementing the technique correctly, not the motivating constraint.

---

## ARM Cortex-M4 cross-compilation

Compiled with `arm-none-eabi-gcc`, targeting `-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16` (hardware FPU enabled, not software float emulation).

| Section | Size       | Meaning                                  |
|---------|------------|--------------------------------------------|
| `.text` | 5,040 bytes | Compiled code + weight constants (flash)  |
| `.data` | 0 bytes     | No non-const initialized globals          |
| `.bss`  | 508 bytes   | Runtime RAM for activation buffers        |

This measurement is the **deployment build** — compiled with the 300-sample validation set excluded, since that data is test scaffolding and would never ship on a real device, which reads one MFCC vector at a time from a live microphone rather than holding a pre-loaded validation set in flash.

**What this does and does not prove:** the engine compiles cleanly for a real Cortex-M4 target, with zero warnings, using correct hardware-FPU codegen flags. It does **not** prove the code runs correctly on physical hardware — no Cortex-M4 board was available for this project, so nothing here was executed on real silicon. Correctness was instead verified on the host machine (the 81.7%-match result above). It is reasonable to expect this would also run correctly on real hardware, since the code uses no exotic instructions and targets a standard, well-supported core — but "reasonable to expect" is not the same claim as "verified," and this project does not claim hardware verification.

---

## Inference timing

**This number is not representative of embedded hardware and should not be read as one.** It was measured by running the validation loop natively on a MacBook Pro (`gcc`, no ARM cross-compilation, no embedded timing hardware) and is included only to show the loop completes and produces a number, not as a performance claim:

- ~0.02 ms average per inference, host machine, native compile

Cortex-M4 inference time was not measured, since no board was available. A Cortex-M4 typically runs at 80–180 MHz with no out-of-order execution and no SIMD comparable to a modern laptop CPU, so real hardware latency would be substantially higher than the host figure above — plausibly low single-digit milliseconds — but this is an estimate based on general knowledge of the core, not a measurement, and should be presented as such if asked.

---

## What was actually engineered here

- **Per-layer asymmetric quantization** — independent scale and zero-point per weight tensor and per activation, derived from each tensor's actual min/max rather than a single global scale.
- **Float-domain rescaling with bias** — the int32 matmul accumulator (`(W_q - W_zp) · (x_q - x_zp)`) is converted to float, scaled by `x_scale * W_scale`, and combined with a float bias before ReLU and re-quantization. This exact ordering is what makes the C engine's output match the Python reference instead of silently diverging.
- **Hardware FPU explicitly targeted** (`-mfloat-abi=hard -mfpu=fpv4-sp-d16`) rather than left to default software float emulation.

## What was not engineered / explicitly out of scope

- No physical hardware validation (compilation only).
- No temporal feature modeling — MFCC mean collapse is a known, accepted ceiling on accuracy.
- No latency or power measurement on real silicon.