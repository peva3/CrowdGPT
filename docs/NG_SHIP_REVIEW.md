# Ship review — principles vs. what is actually wired

Date: 2026-11-09. Reviewed against the principles NG-1…NG-6 and PM-1…PM-4 in
`TODO-NG.md`. This is a gap list, not a certification. Each item names the code
that exists, what is missing, and why it matters to a principle.

Severity: **A** = fix before enabling the feature in a real pool;
**B** = ship with the upgrade or the claim is overstated;
**C** = next iteration.

## A — correct before turning LoRA on

### A1. LoRA and dense clients can share a pool (NG-4)

`_hw_signature()` (`server.py:1257`) builds the pool key from backend, batch
bucket, autocast, optimizer, GPU class, and compute capability. It has no
train-mode bit. The client has no `X-Client-Train-Mode` header
(`_hardware_headers` in `client.py`). LoRA is enabled only by the local
`CROWDGPT_LORA` environment variable.

Consequence: a rank-16 LoRA client and a full-parameter client can be matched
into one pool. The LoRA submission is expanded to a dense delta server-side,
then the mandatory cross-check compares a low-rank direction against a full
update. One side is likely to be flagged and discarded, or the pool is
discarded entirely. It is a silent throughput and fairness bug, not a crash.

Fix: advertise train mode, add it to the hardware signature as an exact-match
element, put `trainMode` and `loraRank` in the task payload, reject a submit
whose declared mode differs from the ticket, and add a mixed-pool test. Do not
enable `CROWDGPT_LORA` in any shared pool before this.

### A2. LoRA turns engram training off without saying so (PM-2, NG-6)

`client.py` skips the engram sparse delta when `LORA_ACTIVE` is set. That is a
deliberate simplification (the server rejects LoRA + engram), but it changes
what the published checkpoint contains. A LoRA round and a dense round train
different parameter sets, and the record does not say which.

Fix: choose explicitly — either LoRA trains adapters only and the record says
`trained: [adapters]`, or LoRA also trains engram rows and the server accepts
that combination. Either way the nutrition label must report it.

## B — the current claims outrun the wiring

### B1. A round is recorded but not evaluated (NG-4, PM-2)

`_publish_round_record()` writes `eval=None` (`server.py:1879`). The held-out
harness from P0.3 exists (`crowdgpt_ng/eval.py`) but is not called by the
coordinator. So a signed record proves who published the round and what
submissions it contained; it does not prove the model got better.

Fix: run a pinned, sampled eval after aggregation, store loss + CI + manifest
hash in the record and in a per-checkpoint label. Until then, describe records
as "attested", not "evaluated".

### B2. There is no nutrition label artifact (PM-2)

The audit page shows round, leader, roots, and bytes. There is no artifact next
to `model_bf16.safetensors` that a person can read: base revision, code
version, training tokens, contributor count, eval result, known limitations.

Fix: emit `nutrition.json` in the HF upload folder next to the weights, derived
from the signed record, and link it from the audit page.

### B3. Leaderboard is not derived from records (PM-4, NG-6)

`update_and_save_status()` ranks `users.json` by self-reported `totalTokens`
(`server.py:2237`). Signed submissions and receipts exist in SQLite, but they
do not drive the leaderboard. A participant can rank without a verified
submission, and two devices of one account are not distinguished.

Fix: build rankings from `submissions` + `receipts` (verified tokens, accepted
rounds, streak) and keep the legacy number as a labelled fallback during
migration.

### B4. Metrics are a library, not a feed (X5)

`orchestrator/metrics.py` and its three tests are real. `server.py` never
constructs a `MetricsRecorder`; nothing writes a `RoundMetric`. The
observability doc describes rows that do not exist yet.

Fix: write one row per round close, plus alert thresholds. Alternatively
correct the docs to say the recorder is available but unwired.

### B5. Client failover covers POST only (P3.4, PM-1)

`_post_json_with_retry()` rotates base URLs. `wait_for_round()`
(`client.py:2189`) and `fetch_task_metadata()` (`client.py:2243`) call
`requests.get(f"{srv}/...")` directly. A client whose node dies while it is
waiting never moves to the next orchestrator.

Fix: route every request through one URL-rotating transport (including GET and
streamed weight downloads), and pin a node per round only after a ticket is
issued.

### B6. The 1-bit codec is not on the wire (P5.1)

`orchestrator/compression.py` compresses and decompresses, and the tests pass.
Neither the client upload nor the server decode path calls it. Dense pools
still move ~1.3 GB per unit; only LoRA pools are small.

Fix: add a `deltaCodec` field, accept `ngd1` on submit, and measure one real
delta fixture rather than a synthetic 1M-param array.

### B7. Protocol is implemented twice (NG-5)

`crowdgpt_ng/protocol.py` and `orchestrator/records.py` both define canonical
JSON, Merkle roots, and signing. The server duplication is deliberate (it must
be deployable alone), but nothing asserts the two produce identical bytes.

Fix: a conformance test that builds the same Ticket/Receipt/RoundRecord from
both modules and compares canonical bytes, hashes, and signatures.

## C — structural gaps that need a project, not a patch

### C1. Quorum primitives do not make a quorum

Each orchestrator has its own SQLite database. `LeaseManager` and
`ClusterView` read local tables. Nothing replicates leases, tickets, or
registries between nodes. `/api/cluster` shows the local view only. A 2-of-3
certificate therefore cannot be assembled across real nodes yet, and two nodes
can both believe they hold the lease.

This is the largest gap between P3's code and PM-1/NG-3. A real design needs
either a small replicated log (Raft-lite for leases + registry) or an external
coordination service, and it must be measured on three machines. Until then the
honest description is "quorum building blocks", not "quorum".

### C2. `krum_stream` is private, so a public operator cannot aggregate

The fork boots with stubs only. Cross-check, merge, and aggregation import
`krum_stream`, which is not in the repository and not published. An external
operator cannot run a real node from the fork alone.

Fix: publish a reference implementation (even a slow, correct one) or declare
it an explicit dependency with a pinned distribution.

### C3. No public-record verification in the client

The orchestrator signs records; the client never verifies a certificate or a
Merkle proof. A malicious or forked orchestrator can present a different chain
to a client with no detection. Client-side verification is what makes
"anyone can audit" real for participants, not just for a website visitor.

### C4. No correction/errata path for published records

`publications.round` is a primary key and the first write wins. If a record is
wrong (evaluator bug, bad merge), there is no signed errata object. Add a
supersede/errata record type with its own signature and a pointer from the
original.

## Lower-priority hardening

- Admin endpoints take the secret as a query parameter (`server.py:4020` and
  neighbours). Query strings land in proxy and access logs. Accept an
  `X-Admin-Secret` header and deprecate the query form.
- `slowapi` is stubbed in tests and unpinned in `requirements.lock`; pin it or
  make the limiter optional in the public build.
- Ticket TTL is a fixed 900 s. At the measured 1.037 s/step (seq 2048), 500
  steps is ~8.6 min and fits; a slower card will expire mid-unit. Derive the
  TTL from measured tokens/s or extend it on a progress heartbeat.
- Round length is still 4 h while units are ten minutes. Decide whether
  aggregation runs every N units or rounds stay long with short tickets.
- No backup/restore rehearsal for `orchestrator_state.sqlite3`; the runbook is
  text only.
- No SBOM or reproducible-build check (documented as a known limitation).

## Suggested order

1. **A1 + A2** before any LoRA pool is enabled.
2. **B1 + B2** so a published checkpoint carries an actual evaluation.
3. **B5** because failover is a headline claim and currently half-true.
4. **B4 + B7** are small and make existing claims accurate.
5. **C1 + C2** are the real gate to a permissionless pilot; C1 is the larger
   engineering item.
6. **C3 + C4** before inviting third-party auditors.
7. **B3 + C4** as the participation/rewards layer matures.
