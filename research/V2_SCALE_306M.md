# Scale check: backprop vs MOSAIC at 306 M

The tiny-model table in `V2_PARADIGMS.md` says MOSAIC (2 slices, H=5) beats
backprop at 40 rounds. That is a wiring result, not a scaling result, so this is
the first measurement at a size worth quoting: 306 M parameters
(V=4096, D=1024, L=24, H=16, T=512, B=4, tied readout, GPT-2 0.02 init), 8 fixed
`val.bin` windows, both arms to a 2 M-token budget.

Harness: `experiments/v2/scale_306m.py`.

## Result (CUDA, RTX 4060 Ti 16 GB)

| arm | init | ¼ | ½ | final | wall | cost |
|---|---|---|---|---|---|---|
| backprop | 8.5367 | 6.2236 | 5.9203 | **5.6862** | 478.0 s | 4,181 tok/s |
| MOSAIC K=2 H=30 | 8.5367 | 7.8544 | 7.7257 | **7.7166** | 331.6 s | see below |
| MOSAIC K=2 H=5 | 8.5367 | 7.4897 | 7.4125 | **7.4921** | 347.4 s | 5,719 tok/s |

`delta(MOSAIC - backprop) = +2.0303` at H=30 and `+1.8059` at H=5. MOSAIC is not
close on this budget — but read the next section before concluding anything.

Lowering H from 30 to 5 halves the gap and leaves the same pathology, which is
the signature of the *budget*, not of the outer rule: at `--budget inner` a
smaller H buys more rounds and more unique data (97 rounds, 397,312 unique
tokens at H=5, versus 16 rounds and 65,536 at H=30), so quality tracks data seen
rather than the local-step count.

## Why that number is not yet a verdict on MOSAIC

The two arms do not consume the same data, and the difference is not small.

`per_round = slices × local_steps × B × T = 2 × 30 × 4 × 512 = 122,880`
inner-step tokens, so a 2 M-token budget buys **16 outer rounds**. Each slice
spends those 30 steps on **one** batch, so the data actually seen is
`16 × 2 × 4 × 512 = 65,536` unique tokens — **30× less than backprop's 2 M** — for
960 gradient steps instead of 976.

The reported 5,930 tok/s is inner-step throughput; unique-token throughput is
65,536 / 331.6 = **198 tok/s**. The arm trained on 1/30 of the corpus and still
finished 1.4× faster in wall clock, which is exactly what a data-starved run looks
like.

So the measurement to trust so far is narrow: *at a fixed budget of inner-step
tokens, H=30 leaves too few aggregation rounds and too little data*. It does not
say MOSAIC's clipping/averaging rule is wrong.

## The harness now reports the distinction

`--budget` chooses how `--tokens` is spent:

- `unique` (default) — equal data: `rounds = tokens / (slices·B·T)`, so H local
  steps cost proportionally more compute (the method's price) and nothing is
  starved. This is the fair test.
- `inner` — equal inner-step tokens, i.e. the run above, kept only to reproduce.

Both arms now report `gradSteps`, `rounds` and `uniqueTokens` alongside `tokens`,
so an equal-data claim cannot be read off an equal-inner-token run.

At 2 M tokens, `unique` budgeting gives 488 rounds for every H:

| H | rounds | gradient steps | vs backprop (976) |
|---|---|---|---|
| 1 | 488 | 976 | 1× |
| 5 | 488 | 4,880 | 5× |
| 30 | 488 | 29,280 | 30× |

H=1 is the decisive cheap probe: identical data, identical step count, only the
outer rule (median clip, slice averaging, `outer_lr=0.7`, momentum 0.9) and the
SOLO head on slice 0 differ. If H=1 loses to backprop, the rule is the problem;
if it matches, the local-step data reuse is.

## The fair comparison: equal data, equal gradient steps

`--budget unique`, 2 M tokens, 488 rounds for every H. Backprop's reference row is
the same 2 M unique tokens and its 976 AdamW steps.

| arm | ¼ | ½ | final | wall | unique tokens | gradient steps |
|---|---|---|---|---|---|---|
| backprop | 6.2236 | 5.9203 | **5.6862** | 478.0 s | 1,998,848 | 976 |
| MOSAIC K=2 H=1 | 7.2958 | 7.2624 | **7.2270** | 402.1 s | 1,998,848 | 976 |
| MOSAIC K=2 H=5 | 7.3318 | 7.2714 | **7.2285** | 1734.5 s | 1,998,848 | 4,880 |

At H=1 the arms see identical data and take the identical number of gradient
steps; only the outer rule differs (median clip, mean of slice deltas,
`outer_lr=0.7`, momentum 0.9) plus the fact that slice 0 trains against a SOLO
head. MOSAIC ends **1.54 nats behind** (7.2270 vs 5.6862). That is the answer the
inner-budget runs could not give: the starvation was real, but removing it does
not close the gap.

Five times the local steps on the same data (H=5 — 4,880 gradient steps, 3.6×
wall clock) lands within 0.0015 of H=1, i.e. more local work bought nothing, and
both MOSAIC curves flatten between ½ and final (7.2624 → 7.2270, 7.2714 → 7.2285)
while backprop is still descending (5.9203 → 5.6862). A curve that plateaus that
hard is not merely slower; at this slice count, inner lr and outer lr the outer
loop is not converging.

Caveat, stated rather than buried: K=2 means 12 layers per slice, and slice 0
trains against a SOLO head initialised at `attach` time, so over 2 M tokens the
head has little chance to align with the readout. More data, more slices, or a
warm head could look different. What is measured is the regime a weekend launch
would actually start in.

## What this means for a launch

- **Backprop is the arm to start a new model with** at this scale.
- MOSAIC is *verified as wired* — delta layout, slice integrity, both devices —
  and *not verified as competitive* at 306 M / 2 M tokens.
- The 40-round tiny-model win in `V2_PARADIGMS.md` is a wiring result; it does not
  survive a fair 306 M comparison.

## Status

- CUDA H=30 and H=5 (`--budget inner`): measured — artifacts of the budget, kept
  only to show why they looked bad.
- CUDA H=1 and H=5 (`--budget unique`): measured — the fair comparison above.
- ROCm 890M (`--budget inner`): still running; will be recorded as another
  inner-budget artifact, not as a verdict.

Numbers land here as they finish; the table is updated rather than replaced so the
correction stays visible.
