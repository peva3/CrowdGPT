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
