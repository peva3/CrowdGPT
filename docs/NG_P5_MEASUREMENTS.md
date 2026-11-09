# P5 measurements — LoRA, checkpointing, DiLoCo

Date: 2026-11-09. Scope: single RTX 4060 Ti 16 GB, random weights, synthetic
quadratic for DiLoCo. These are mechanism and memory measurements, not a
production quality evaluation.

## LoRA on the full model

Command:

    .venv/bin/python tools/bench_lora_gpu.py --seq 512 --checkpoint on --steps 3
    .venv/bin/python tools/bench_lora_gpu.py --seq 512 --checkpoint off --steps 3
    .venv/bin/python tools/bench_lora_gpu.py --seq 2048 --checkpoint on --steps 2

Measured:

| seq | checkpoint | peak GB | s/step | tok/s |
|-----|------------|---------|--------|-------|
| 512 | on | 3.54 | 0.3134 | 1633.8 |
| 512 | off | 5.01 | 0.2271 | 2254.9 |
| 2048 | on | 3.79 | 1.0368 | 1975.2 |

The model has 657,787,392 base parameters. The rank-16 adapters total
8,552,448 trainable parameters. A full-model adapter payload is 18.88 MB in
the `NGL1` wire format, versus a dense 1.316 GB base delta.

Interpretation: an 8 GB card has enough memory for a seq-512 unit with
checkpointing at this batch size; a 12 GB card has headroom for seq 2048. This
is a memory-fit result, not a held-out loss comparison.

## Activation checkpointing

The same probe, seq 512 with checkpointing on vs off, shows a 1.35–1.38×
throughput gain when checkpointing can be safely disabled. A tiny CPU
microbenchmark (`tools/bench_model_paths.py`) shows the same direction
(1.82×) and a forward-loss delta of 0.00000000, which checks that the switch
does not change the math.

## LoRA vs full-parameter on a short held-out probe

Command:

    .venv/bin/python tools/bench_update_tracks.py --steps 30 --seq 128

Random-initialised 658M model, repository text, 3,840 training tokens, the same
four held-out windows. Measured on the RTX 4060 Ti 16 GB:

| arm | trainable params | before | after | peak GB |
|---|---|---|---|---|
| LoRA rank 16, AdamW 1e-3 | 8,552,448 | 12.1004 | 10.9054 | 3.39 |
| full, SGD 1e-4 | 657,788,928 | 12.1004 | 12.0952 | 6.69 |

This is a 30-step mechanism probe, not evidence that LoRA is generally better
than full-parameter training. The full arm used a lower learning rate after
1e-3 diverged. Report it only with those qualifications.

## DiLoCo outer loop

Command:

    .venv/bin/python tools/bench_outer_loop.py --rounds 30

Synthetic quadratic, 8 workers × 20 local steps × 30 rounds = 4,800 gradient
steps per arm. Per-round averaging reaches 2.745401; DiLoCo with persistent
Nesterov momentum reaches 2.745767. The compressed 8-float pseudogradient is
23 bytes (0.1437 bytes/token at 160 tokens/round). This checks convergence and
compression mechanics at equal work, not production model loss.

## What remains unmeasured

- A real trained checkpoint's held-out loss under LoRA vs full-parameter
  updates.
- A production round's per-client wall clock and bytes.
- Multi-GPU / multi-node throughput.

No speedup or quality claim beyond the tables above should be inferred.

## 1-bit delta codec on the wire

`CROWDGPT_DELTA_CODEC=ngd1` writes a framed 1-bit container (`NGD2`): each
frame is `NGD1` + per-1024 fp16 mean-abs scale + sign bits + zlib. The server
decodes frames into the dense bf16 stream, so aggregation and cross-check are
unchanged. LoRA and engram tails keep the existing paths; the codec is used for
dense, base-only deltas.

Synthetic 4,000,000-param fixture (IEEE float32, sigma 0.01):

| measure | value |
|---|---|
| dense bytes | 16,000,000 |
| compressed bytes | 505,814 |
| ratio | 31.6x |
| bytes per param | 0.12645 |
| projected 657,787,392 params | 83.2 MB |
| mean abs error | 0.004823 |
| mean abs value | 0.007975 |

The projection is arithmetic, not a captured production delta; a real delta
fixture is still unmeasured. Reconstruction is intentionally coarse (one sign
plus a block scale), so this is a bandwidth mechanism, not a lossless path.

## NGD1 codec on a local model delta

Command:

    .venv/bin/python tools/bench_codec_fixture.py --params 64000000

One full-parameter SGD step on the 658M model (random tokens, seq 128), then
the first 64M parameters of the update:

| measure | value |
|---|---|
| dense bytes | 256,000,000 |
| compressed bytes | 1,937,282 |
| ratio | 132.14x |
| bytes/param | 0.03027 |
| encode / decode | 0.771 s / 0.710 s |
| mean abs error | 1.156e-08 |
| mean abs value | 1.331e-08 |

This is a **local model delta**, not a captured production client delta. The
small SGD step produces very small values, and the 1-bit sign codec's error is
the same order as the update magnitude. The ratio is real; the reconstruction
error is not evidence that the codec preserves a production update.
