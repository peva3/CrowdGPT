# NG server base — CrowdGPT-Server

Recorded: 2026-11-09
Repository: `https://github.com/Vxtzq/CrowdGPT-Server`
Local checkout: `/app/crowdgpt/CrowdGPT-Server`
Base commit: `34f190bb722e2de55de7473b5f34f862c806cad6`
Commit time: 2026-11-07T18:29:54+01:00
License: GPL-3.0

This is the upstream coordinator (the README calls it the *coordinator*; the ng
fork calls a permissionless instance an *orchestrator node*). It is a separate
repository from the client and is **not vendored** into `Crowdgpt-ng`.

## What the repository contains

```
LICENSE
README.md
public_log.py   (8.8 KB)
server.py       (213 KB)
```

There is no test suite and no packaging metadata. `server.py` is one FastAPI
application (`app = FastAPI(...)`), with `uvicorn` started only under
`__main__`. All data paths are relative to `Path(__file__).parent`:
`checkpoints/`, `data/`, `users.json`, `pending_uploads/`, `work_tmp/`.

## Missing deployment dependencies

The public tree cannot be imported as-is:

- `krum_stream` is imported at `server.py:132` but is not in the repository and
  is not a published package. It provides `open_delta`,
  `streaming_pairwise_sq_l2`, `stream_add_base_delta`, `read_engram_tail`, and
  `plan_round`. Cross-check, merge, and aggregation depend on it.
- `slowapi` is listed in the README but is not installed in the workspace
  virtualenv. It provides the rate limiter.

`tests/server_stubs/` in this repository contains small stand-ins used only by
the offline harness. They are not a production implementation and must not be
deployed.

## Offline test harness

`tests/test_ng_server_harness.py` copies `server.py` and `public_log.py` into a
temporary directory, puts `tests/server_stubs/` first on `sys.path`, imports
the module under a unique name, and shrinks the model so the real upload code
runs on CPU:

| Setting | Value |
|---|---|
| `MODEL_SIZE` | 64 |
| `ENG_PARAMS` | 0 |
| `ENG_NUM_BUCKETS` | 4 |
| `MODEL_CONFIG["dim"]` / `["vocabSize"]` | 4 / 8 |
| `MATCHMAKING["POOL_SIZE"]` | 2 |
| `HARDEN["MIN_TRAIN_TIME_S"]` / `["MIN_SUBMIT_INTERVAL_S"]` | 0 / 0 |
| `HARDEN["TRUSTED_PROXY_NETS"]` | `[]` (tests identify clients with `X-Forwarded-For`) |

The harness does not run the FastAPI startup event. It sets `dataset_info` and
`data_position` directly, installs zero weights, and drives `TestClient`
without a context manager.

Run it with:

```bash
cd Crowdgpt-ng
CROWDGPT_SERVER_DIR=/app/crowdgpt/CrowdGPT-Server \
  /app/crowdgpt/.venv/bin/python -m pytest tests/test_ng_server_harness.py -q
```

Current result: **2 passed**.

The tests cover:

1. `/fl/round_status`, two-client pool formation, re-match of the waiting
   client to the same task, `/fl/submit/init`, `/fl/submit/chunk`,
   `/fl/submit/finalize`, and `/api/admin/force_round_end`.
2. A simulated restart: a second module instance over the same files starts
   with empty tickets, tasks, pools, pool map, and ready-pool list.

## Baseline bytes per orchestrated round

Derived from the server constants, not yet measured end-to-end on a real
client (per-client timing and byte counts are not published today):

| Item | Bytes |
|---|---|
| Base weights, bf16 (657,787,392 params) | 1,315,574,784 |
| Engram weights, bf16 (350,000,640 params) | 700,001,280 |
| Full `/fl/task` download (base + engram) | **2,015,576,064** (~1.877 GiB) |
| Base delta upload | 1,315,574,784 |
| Engram tail, worst case (all buckets) | 700,001,280 |
| Base-only round, down + up | ~2.63 GB |
| Worst-case full round, down + up | ~4.03 GB |

A real sparse engram tail is smaller than the worst case. The previously
observed delta size was about 1.386 GB per round, consistent with the base
delta plus a sparse tail.

Client upload chunk size is 64 MiB (`UPLOAD_CHUNK_BYTES` in
`CrowdGPT/client.py`), halved on HTTP 413 down to 8 MiB. A base-only upload is
20 chunks; a full-engram upload is 31. One submitted client therefore makes:

- 1 `/fl/round_status`
- 1–2 `/fl/task` polls
- 1 `/fl/submit/init`
- 20–31 `/fl/submit/chunk`
- 1 `/fl/submit/finalize`

≈ 24–36 HTTP requests, plus heartbeats. The checked-in harness records 8
requests for one client's partial round including the admin call. A 3-client
pool is roughly 70–105 requests; six pools are roughly 420–630.

## What a restart loses

`server.py` keeps the round in memory:

- `task_tickets` (issued tickets)
- `db["tasks"]` (task metadata)
- `pools`, `client_pool_map`, `ready_pools`
- `round_state` (including `force_end_at`)
- `active_clients`, `client_hardware`
- submission quota counters and last-submit times
- `byzantine_flag_counts`, subnet bans, user bans, banned client keys

The startup event clears pools and ready pools, deletes pending delta files and
upload directories, and starts a new round at `global_step + 1`. The README is
explicit: *"Tasks and pools are kept in memory only: a restart drops every task
in progress."*

Durable state is limited to `checkpoints/global_state.json`,
`checkpoints/model_latest.bin`, `checkpoints/engram_latest.bin`, `users.json`,
`pending_deltas/`, and `pending_uploads/` (6-hour TTL).

This is the reason P1 exists: tickets, round state, and pool membership must be
persisted before an orchestrator can be replaced without losing or duplicating
work.

## Repository state

A writable fork now exists: `https://github.com/peva3/CrowdGPT-Server`
(`fork` remote in the local checkout). It was created 2026-11-09 from
`Vxtzq/CrowdGPT-Server` and fast-forwarded to `f8e445e`, which includes all
P0.4–P5 server work. `origin` still points at the upstream repository, so
push to `fork` explicitly.

The ng changes live in `/app/crowdgpt/Crowdgpt-ng` and are pushed to
`origin/ng/main`.
