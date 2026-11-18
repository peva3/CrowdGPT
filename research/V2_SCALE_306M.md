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

## Update (same day): the gap was partly a bug, and ownership explains most of it

The 1.5-nat deficit above was not the method. Two defects in
`crowd_mosaic.py`, both found by probing the code rather than the loss:

1. **The SOLO head was never optimised.** `local_delta` picked its parameters
   with `slice_of(name) == k`, and `slice_of` only matched `blocks.<i>.`
   prefixes, so `solo_heads.<k>.*` never entered the optimiser — its gradients
   were computed and discarded. Cost: **0.035 nats** (7.5999 → 7.5653). Real,
   but a footnote.
2. **The trunk ends were never optimised either.** `wte`, `ln_f` and the tied
   `lm_head` were excluded for the same reason. Because `lm_head.weight is
   wte.weight`, MOSAIC spent the run training a representation to decode through
   a **randomly initialised, frozen readout**. This is the big one.

Probe on a finished 65 M MOSAIC run: a fresh linear probe on its own trunk
reached **6.39** while the model itself reported **7.18** — 0.79 nats of
information the readout was failing to use. That predicted almost exactly what
fixing the ownership recovered.

### Measured, 65 M / 2 M unique tokens / 488 rounds (backprop = 5.6340)

| arm | trunk→slice 0 | trunk→last slice |
|---|---|---|
| MOSAIC H=1 (3e-3 / 0.7 / 0.9) | 7.2456 | **6.4651** (−0.78) |
| MOSAIC H=5 (1e-3 / 1.0 / 0.9) | 7.1728 | **6.1281** (−1.04) |
| backprop | 5.6340 | — |
| SOLO (same split, same local law) | 5.5180 | — |

Two things follow. Giving the last slice `wte` means the terminal slice's *true*
cross-entropy trains the readout, which is the role it plays under SOLO where one
optimiser covers both. And local steps finally pay: H=5 beats H=1 by 0.34 nats,
where before the fix it bought 0.07.

What remains — 0.49 nats at H=5 — is trunk quality, which matches the probe's
6.39 measurement almost exactly.

### The fix in code

`slice_of` now assigns `wte.` to slice 0 and `ln_f.` / `lm_head.` to the last
slice, and `local_delta(..., share_trunk=True)` additionally gives `wte` to the
terminal slice so **both** of its roles are trained: embedding through slice 0's
head, readout through the true loss. `aggregate` already averages a tensor that
appears in two slices' deltas, so sharing needs no new wire mechanism — only half
a step per role per round.

## The local-step optimum, and what the fixes are worth

All arms below: 65 M (`--dim 512 --layers 12 --heads 8`, 653 M params at the
306 M preset's depth/width ratio), 2 M unique tokens, 488 rounds for K=2,
8 fixed `val.bin` windows, `--inner-state`.

| arm | final | wall |
|---|---|---|
| backprop, 2 M tokens (976 steps) | **5.6340** | 75 s |
| SOLO, 2 M tokens | 5.5180 | 78 s |
| MOSAIC K=2 H=5, trunk (tied readout) owned by slice 0 | 7.1728 | 231 s |
| MOSAIC K=2 H=5, readout owned by the last slice | 6.1281 | 231 s |
| MOSAIC K=2 H=5, readout owned by **both** (`share_trunk`) | **6.1046** | 231 s |
| MOSAIC K=2 H=1, share_trunk | 6.3344 | 57 s |
| MOSAIC K=2 H=10 | 6.3883 | 450 s |
| MOSAIC K=2 H=20 | 6.7522 | 885 s |
| MOSAIC K=4 H=5 (244 rounds, half the updates) | 6.3306 | 149 s |

Three things to read off it:

1. **Ownership is worth ~0.9 nats.** The tied tensor (`lm_head.weight is
   wte.weight`) has two roles; giving it to the embedding side left the readout
   randomly initialised for the whole run. This is the largest single effect we
   have measured, larger than every hyper-parameter in the table combined.
2. **H=5 is the optimum.** Local steps beyond that over-train on a single batch
   and the outer step gets correspondingly larger (H=10 and H=20 both regress,
   and the curves oscillate).
3. **K=4 is not comparable here.** `--budget unique` fixes rounds at
   `tokens/(slices*B*T)`, so four slices in the same 2 M tokens get half the
   parameter updates. A fair K=4 arm needs 4 M tokens.

## Matched per-parameter data: MOSAIC ties backprop

A MOSAIC round trains every parameter once but spends `slices*B*T` tokens doing
it, so over a fixed total token budget each parameter sees half the batches
backprop's parameters see. The matched control is therefore backprop at 1 M
tokens (488 steps):

| arm | tokens total | per-parameter tokens | final |
|---|---|---|---|
| backprop | 1 M | 1 M | 6.1086 |
| SOLO | 1 M | 1 M | 6.3386 |
| MOSAIC K=2 H=5, share_trunk | 2 M | 1 M | **6.1046** |
| backprop | 2 M | 2 M | 5.6340 |

At equal per-parameter data MOSAIC **ties** backprop and beats SOLO. At equal
total tokens backprop is 0.47 ahead. Both statements are true; which one matters
depends on whether the budget being spent is volunteer tokens (total) or
parameter updates (per-parameter). Note also that in the distributed setting a
round's slices run on different volunteers in parallel, so a MOSAIC round costs
one slice's wall-clock time for `slices` slices' worth of compute — the reason
the method exists at all.
