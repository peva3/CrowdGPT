# NG baseline — what the current network actually does

Recorded: 2026-11-09
Sources: `data/rounds.jsonl` (backfilled from the 2026-11-09T00:23:39Z
snapshot), `data/summary.json`, `CrowdGPT-Server/server.py` constants, and
`CrowdGPT-ng/client.py`.

This is a **before** measurement. It is not a performance claim about a new
build, and it does not turn reported numbers into checked ones.

## Round cadence (measured)

52 closed legacy rounds, 286–337:

| Measure | Value |
|---|---|
| Time span | 2026-10-30T12:28 → 2026-11-08T19:49 (9.3 days) |
| Rounds per day | 5.59 |
| Round gap, mean | 4.38 h |
| Round gap, p50 | 4.10 h |
| Round gap, p95 | 5.51 h |
| Longest gap | 17.13 h |

The configured round duration is 4 h, so cadence is close to the target.

## Work per round (measured, legacy records)

| Measure | p50 | p95 | Mean |
|---|---|---|---|
| Tokens per round | 340,408,320 | 477,673,267 | 302,172,396 |
| Deltas per round | 92 | 116 | 83.1 |
| Tokens per delta | 3,939,894 | 5,133,015 | 3,658,793 |

Total tokens in the window: 15,712,964,608 (15.71B). This is the sum of the
reported per-round token counts; it is not a held-out or verified count.

## Bytes per unit (derived from server constants)

| Item | Bytes |
|---|---|
| Base weights, bf16 | 1,315,574,784 |
| Engram weights, bf16 | 700,001,280 |
| Full weight download | 2,015,576,064 (~1.877 GiB) |
| Base delta upload | 1,315,574,784 |
| Observed delta (base + sparse engram) | ~1.386 GB |
| Base-only round, down + up | ~2.63 GB |
| Observed-delta round, down + up | ~3.40 GB |

At the p50 of 3.94M tokens per delta, the base-only path is about **668 bytes
per accepted token**; the observed-delta path is about **863 bytes per
accepted token**. Both numbers are bandwidth, not compute.

## Requests per unit (derived)

One submitting client uses 1 `/fl/round_status`, 1–2 `/fl/task` polls, 1
`/fl/submit/init`, 20–31 `/fl/submit/chunk` (64 MiB chunks), and 1
`/fl/submit/finalize`: roughly 24–36 requests plus heartbeats. The offline
harness records 8 requests for a one-client partial round including the admin
call.

## Loss provenance

All loss numbers below are **reported by clients or the coordinator**, not
held out and not independently checked.

| Measure | Value |
|---|---|
| Rounds with a reported loss | 52 |
| Mean | 3.0836 nats |
| p50 | 3.1168 nats |
| p95 | 3.4941 nats |
| Min / max | 1.8629 / 3.6600 nats |
| First 8 rounds vs last 8 | 2.7256 → 3.1255 nats |

Two values exist for round 337 and they disagree:

- `status.json` `lastLoss`: **4.144910**
- `loss_history.json` round 337 `loss`: **2.944640**

Both are labelled as loss by different parts of the system. Neither is a
fixed held-out evaluation. They must not be averaged, summed, or compared
across releases. The historical integrity check in
`CrowdGPT/docs/CROWDGPT_GPU_LOSS_CHECK.md` records a revision where the
reported `lastLoss` was 0.079 while measured held-out loss was 11.10 nats on a
code chunk and 7.80 nats on English README text (uniform baseline
`ln(151669) = 11.93`).

## What is still unknown

- **Per-client unit wall clock (p50/p95)**: not published by the coordinator.
  The P0.5 timing log now records queue, weights, train, upload, and total per
  round when `CROWDGPT_TIMING_LOG` is set, but no production samples have been
  collected yet.
- **Per-phase split** for real clients: download vs pool wait vs training vs
  upload. Only the total round cadence is observable today.
- **Verified token counts**: the legacy records carry reported totals only.
- **Held-out loss per revision**: no fixed evaluation set exists yet (P0.3
  provides the harness; it is not wired to the coordinator).
- **Participation conversion**: Hugging Face reports roughly 78k downloads per
  month for `Vxtzq/Crowd-v1`, while the status snapshot shows 92 registered
  participants and 18–19 active GPUs. The download-to-contribution rate is not
  measured beyond that ratio.

## How to reproduce

```bash
cd Crowdgpt-ng
/app/crowdgpt/.venv/bin/python -m crowdgpt_ng.records \
  --status /tmp/status.json \
  --loss-history /tmp/loss_history.json \
  --leaderboard /tmp/leaderboard.json \
  --out data/rounds.jsonl \
  --summary data/summary.json
```

The snapshot used here is dated 2026-11-09T00:23:39Z. `data/summary.json`
records the exact leaderboard share statistics for that snapshot.
