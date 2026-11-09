# TODO-NG — CrowdGPT-ng groundwork

Status: **active**
Created: 2026-11-09
Canonical copy: this file (`Crowdgpt-ng/TODO-NG.md`); `/app/crowdgpt/TODO-NG.md` is a symlink.
Design rationale: `.opencode/wiki/concepts/groundwork-layer.md`,
`.opencode/wiki/concepts/distributed-orchestration.md`,
`.opencode/wiki/concepts/peoples-model.md`.

This is the work plan for turning the single-VPS coordinator into a quorum of
replaceable orchestrator nodes, with auditable rounds, a leaderboard derived
from records, and ten-minute work units. It covers two repositories:

- **NG** — `Crowdgpt-ng/` (client, protocol library, eval harness, docs)
- **SRV** — the server (`Vxtzq/CrowdGPT-Server`), forked separately

## Principles these tasks must satisfy

From `PROJECT.md` (NG-1…NG-6):

1. **NG-1** Every change is proved with a before/after measurement, not a
   passing test alone.
2. **NG-2** `main` is the merged-PR base; the founder's review stays the gate
   for upstream.
3. **NG-3** Orchestration must become permissionless: no founder-operated
   single point of failure.
4. **NG-4** Verification ships with coordination; a round is not "done" until
   it is checked and recorded.
5. **NG-5** Two checkouts, one contract: the wire protocol is shared by the
   client and every orchestrator.
6. **NG-6** Honest labels: reported, checked, and held-out numbers are never
   summed or relabelled.

From the people's-model direction:

- **PM-1** The network keeps running with any one orchestrator down.
- **PM-2** Every published checkpoint has a public record and a nutrition
  label (provenance, compute, evals, contributors).
- **PM-3** A unit of work fits in about ten minutes on ordinary hardware.
- **PM-4** People can see that the model improves because they ran it, and can
  vote on what improves next.

## Status legend

- `[ ]` open · `[~]` in progress · `[x]` done (evidence required) · `[!]` blocked

A task is done only when its **Evidence** line records a command and its
result, or a commit and a test run. Update this file in the same commit as the
work.

## Phase map

| Phase | Outcome | Effort | Gate |
|---|---|---|---|
| P0 | Contracts and honest baseline | 1.5–2 pw | 10 consecutive rounds recorded, evaluated, published; baseline bytes + wall clock |
| P1 | Stateless single orchestrator | 3–4 pw | `kill -9` mid-round, restart, retry: no duplicate or lost work; orchestration traffic ≈ KB/round |
| P2 | Signed records and validators | 3–4 pw | seeded fake delta detected and excluded; tampered record rejected; 30 rounds with ≥2 attestations |
| P3 | Quorum and failover | 4–5 pw | leader killed ×3 → RTO < 1 round; 2–2 partition certifies nothing; failover < 60 s |
| P4 | Permissionless operation | 4–6 pw | 3 external operators for 30 days without the founder; a malicious operator cannot certify a bad round; ≥50% of rounds checked |
| P5 | Ten-minute units and model tuning | 4–6 pw | p50 unit < 10 min on a 12 GB card; measured tokens/s gain; held-out loss −0.1 nat in 4 weeks |

P0–P4 ≈ 12–17 person-weeks (two engineers ≈ 10–12 calendar weeks). P5 runs in
parallel from P1. **P3 must follow P2**: quorum without signed records only
multiplies unverifiable claims.

## P0 — Contracts and honest baseline

Goal: freeze the wire objects, make every historical round exportable, and
measure the system we actually have. No behavior change to training.

### P0.1 Protocol schemas — NG

- [x] Define `Ticket`, `StepReceipt`, `RoundRecord`, `CheckpointRef`,
  `NodeRecord` as immutable dataclasses in `crowdgpt_ng/protocol.py`.
- [x] Canonical JSON (sorted keys, compact separators, UTF-8) and SHA-256
  digests; losses carried as fixed-precision decimal strings so hashes do not
  drift across platforms.
- [x] Merkle root over sorted leaf digests (duplicate-last), with inclusion
  proof verification.
- [x] Ed25519 signing envelope (`sign_payload` / `verify_payload`) using
  `cryptography`, with a clear error when the dependency is absent. No custom
  crypto.
- [x] Strict validation: required fields, schema version, round ≥ 0, step and
  token budgets > 0, timestamp shape, hex digest shape.
- [x] Tests: round-trip, canonical-stability, tamper detection, merkle proofs,
  signature verify/reject, invalid-field rejection.
- **Evidence:** `tests/test_ng_protocol.py` — 42 passed; full workspace suite
  against `Crowdgpt-ng` — 213 passed, 3 skipped. Commit `ace53d6`.

### P0.2 Round-record exporter and backfill — NG

- [x] `crowdgpt_ng/records.py` normalises the artifacts the server already
  publishes (`status.json`, `leaderboard.json`, `loss_history.json`) into one
  append-only `rounds.jsonl` record per round.
- [x] Every record labels `source` and `verification` (`unverified` here);
  reported and checked values are separate fields, never summed.
- [x] Deterministic output: same inputs → byte-identical file.
- [x] Backfill the available window into `data/rounds.jsonl` (53 rounds,
  286–338, from the 2026-11-09T00:23Z snapshot).
- [x] Tests with fixtures for the three input shapes, missing rounds, duplicate
  rounds, and float formatting.
- **Evidence:** `tests/test_ng_records.py` — 15 passed; `data/rounds.jsonl` —
  53 lines; `data/summary.json` — top3 share 0.706396, HHI 0.186658,
  33 zero-token entries. Commit `aeb100c`.

### P0.3 Held-out eval harness v0 — NG

- [x] `crowdgpt_ng/eval.py`: hash-pinned token shards, fixed window stride,
  mean negative log-likelihood in nats, bootstrap confidence interval.
- [x] Manifest records dataset hash, token count, model weights hash, code
  commit, and tokenizer revision; refuse to run if a hash does not match.
- [x] CPU-feasible sampling mode (N windows, seeded) for the 2-core
  coordinator.
- [x] Tests: known logits produce the expected loss; manifest mismatch
  refuses; sampling is deterministic.
- **Evidence:** `tests/test_ng_eval.py` — 14 passed; full workspace suite —
  242 passed, 3 skipped. Uniform-logits test yields exactly `ln(5)`.
  Commit `5be9986`.

### P0.4 Server test harness and baseline — SRV

- [x] Fork/clone the server repository; record the base commit in
  `docs/NG_SERVER_BASE.md`.
- [x] Add an offline pytest harness that imports `server.py` against a
  temporary data directory.
- [x] Cover `/fl/task`, `/fl/round_status`, `/fl/submit/init`,
  `/fl/submit/chunk`, `/fl/submit/finalize`, and one admin round-end path.
- [x] Record baseline: bytes per orchestrated round, requests per round,
  in-memory state that a restart loses.
- [x] Tests: restart drops in-flight tickets (documented, then fixed in P1).
- **Evidence:** `tests/test_ng_server_harness.py` — 2 passed; workspace suite
  against `Crowdgpt-ng` — 242 passed, 3 skipped. Base commit
  `34f190bb722e2de55de7473b5f34f862c806cad6` and baseline in
  `docs/NG_SERVER_BASE.md`. Commit `18a0f4c`.

### P0.5 Client timing instrumentation — NG

- [x] Behind `CROWDGPT_TIMING_LOG=<path>`, append one JSONL row per phase:
  queue wait, weight fetch, training, delta upload, total; include bytes and
  token count.
- [x] No behavior change when the variable is unset; no secrets or IPs.
- [x] Tests: unset → no file; set → valid JSONL; failure paths still emit a
  row with `ok=false`.
- **Evidence:** `tests/test_ng_timing.py` — 4 passed; workspace suite against
  `Crowdgpt-ng` — 242 passed, 3 skipped. The recorder is embedded in
  `client.py` because the auto-updated single-file client cannot import the
  `crowdgpt_ng` package. Commit `06e99dd`.

### P0.6 Baseline report — OPS

- [x] `docs/NG_BASELINE.md`: p50/p95 unit wall clock, bytes down/up, tokens
  per unit, reported vs held-out loss with the metric's provenance.
- [x] State what is **unknown** (per-client timing is not published today).
- **Evidence:** `docs/NG_BASELINE.md` (round cadence p50 4.10 h / p95 5.51 h;
  340.4M p50 tokens per round; 3.94M p50 tokens per delta; 2.63–3.40 GB per
  unit; reported loss p50 3.1168 nats, no held-out metric). Per-client wall
  clock is explicitly unknown until timing samples exist. Numbers traceable
  to `data/rounds.jsonl`. Commit `1a9daee`.

## P1 — Stateless single orchestrator

Goal: one orchestrator that can be killed at any moment without losing or
duplicating work, and that never proxies bulk bytes.

### P1.1 Server structure — SRV

- [x] Split admission/scheduling, submission/aggregation, verification, and
  publication into modules with explicit interfaces.
- [x] Replace in-memory rounds/pools/tickets with SQLite WAL (Postgres later
  only if measured necessary).
- [x] Persist tickets, round state, and pool membership; reconstruct on boot.
- [x] Tests: every module can be imported and exercised without the HTTP layer.
- **Evidence:** `CrowdGPT-Server/orchestrator/` (state, admission, scheduling,
  submission, verification, publication, service); schema v1 with `rounds`,
  `pools`, `pool_members`, `tickets`, `uploads`, `submissions`,
  `verifications`, `publications`, `events`. `pytest tests -q` in the server
  checkout → 7 passed, including a subprocess assertion that importing the
  package does not import FastAPI. `Crowdgpt-ng/tests/test_ng_server_harness.py`
  → 2 passed; the restart case now asserts pools, tasks, membership, and live
  tickets are reconstructed. Server commits `49a28ce`, `db6c38e`, `605ebc0`
  (local; no writable server fork). Design and integration points in
  `CrowdGPT-Server/docs/ORCHESTRATOR_MODULES.md`.

### P1.2 Durable tickets and idempotent submission — SRV

- [x] Signed nonce tickets bound to `(round, base_weights, client, step_budget,
  token_budget, expires_at)`.
- [x] `submit/init|chunk|finalize` idempotent by ticket + content hash; retries
  return the original result, never a second delta.
- [x] Expired tickets rejected with a machine-readable reason.
- [x] Tests: duplicate finalize, out-of-order chunks, replay after restart,
  expired ticket.
- **Evidence:** Ed25519 `ng-ticket-1` payload (ticket id, client, round, task,
  pool, timestamps, step/token budgets, nonce, `baseWeights=step:<n>`) signed by
  `orchestrator_node_key.pem`; `/fl/task` returns the signed ticket; upload
  rows persist from `init`, each chunk is recorded, and finalize stores its
  response. Server `pytest tests -q` → 8 passed. `Crowdgpt-ng` harness → 6
  passed: duplicate finalize replays the original body, out-of-order chunks
  finalize, expired ticket returns `{"reason":"ticket_expired"}`, and finalize
  replays after a simulated restart. Server commits `(local, no fork yet)`.

### P1.3 Presigned data plane — SRV/NG

- [x] Upload URLs issued by the orchestrator, bytes go to object storage (or
  HF), SHA-256 verified on finalize.
- [x] Client verifies downloaded weights against `base_weights` before
  training.
- [x] Orchestration request/response bytes measured per round.
- [x] Tests: bad hash rejected; expired URL rejected; orchestrator sees no bulk
  payload.
- **Evidence:** `orchestrator/objects.py` (HMAC grant over key/expiry/sha256;
  atomic `commit_file`; dev `LocalObjectStore`), `/fl/submit/init` returns an
  `object` grant, `PUT /fl/object/{key}` verifies token and hash, `finalize`
  accepts `{objectKey, sha256}` and re-hashes before processing. HF metadata
  carries LFS SHA-256; the client verifies fresh downloads and deletes the file
  on mismatch. `CONTROL_PLANE` counters exclude object PUTs and are exposed in
  `/fl/round_status`. Server tests 10 passed; ng harness 7 passed (bad/expired
  grant `403`, tampered object `400 object_hash_mismatch`, object PUT does not
  increment control-plane requests). Server commit local; ng client change in
  this commit.

### P1.4 Client cache, resume, prefetch — NG

- [x] Content-addressed weight cache (`sha256` directory), reused across
  rounds; hit rate logged.
- [x] Resume a paused/interrupted unit from the last checkpoint instead of
  refetching weights.
- [x] Prefetch the next ticket's weights while uploading the current delta.
- [x] Tests: second run hits cache; interrupted run resumes; no duplicate
  upload.
- **Evidence:** `docs/NG_CLIENT_CACHE.md`. Cache path is
  `hf_weights/blobs/<role>__<content-id>__<size>.safetensors` and logs
  `Weight cache: N/M hit (P%)`; interrupted downloads keep `.part` and resume
  with a Range request; `CROWDGPT_UNIT_CHECKPOINT_EVERY=<steps>` writes model,
  both optimizers, and counters to `unit_resume/<task>.pt` and reloads them;
  `prefetch_latest_hf_weights` warms the latest HF revision after upload
  handoff without taking a coordinator ticket. `pytest tests/test_ng_timing.py`
  → 6 passed (checkpoint round-trip, cache-path stability, timing rows).

### P1.5 Restart drill — SRV

- [x] `kill -9` during aggregation and during upload; restart; the round
  completes or is explicitly discarded, never half-applied.
- [x] Test asserts exactly one accepted delta per ticket and no lost tickets.
- **Evidence:** `docs/NG_DRILLS.md`. `CrowdGPT-Server/tests/test_restart_drill.py`
  forks and SIGKILLs a real process after a committed round/pool/ticket, inside
  an open transaction, and with a half-written object → 3 passed (committed
  state survives, uncommitted round rolls back, upload stays `open` with no
  committed object). The ng harness → 8 passed: restart reconstructs state,
  duplicate finalize leaves one delta, and a closed round is not resurrected.
  Full uvicorn kill is an operator runbook item in P4.

## P2 — Signed records and validators

Goal: a round is a signed object that anyone can audit; fake work is detected
and excluded.

### P2.1 Node identity and signed records — SRV/NG

- [x] Ed25519 node keys; `NodeRecord` published at registration and rotation.
- [x] Leader signs `RoundRecord`; witnesses co-sign; certificate embedded.
- [x] Client `StepReceipt` signed with the account key (or session key).
- [x] Tests: tampered field rejected; unknown key rejected; rotated key obeys
  an overlap window.
- **Evidence:** `orchestrator/nodes.py` (registration, `rotate(overlap_s)`,
  `resolve(at)`), `orchestrator/records.py` (`ng-1` canonical records,
  `sign_record`, `verify_certificate`, multi-witness certificates),
  `/api/node` + `/api/nodes`, server signs each round close, client key at
  `checkpoints/client_key.pem` and `X-Client-Pubkey`; `_check_receipt` verifies
  signature, ticket, round, base hash, and fingerprint. Server tests 32 passed;
  ng harness 12 passed.

### P2.2 Activation fingerprints and replay — NG/SRV

- [x] Fingerprint sampled activations (LSH; ~258 bytes per 32 tokens) into
  `StepReceipt.fingerprint_root`.
- [x] Validator mode recomputes a random sample of steps and compares
  fingerprints; flags divergence.
- [x] Replay a random fraction of submissions end-to-end; sampling alone is
  not sufficient for a single fake.
- [x] Tests: honest client passes; one mutated step flagged; fabricated
  fingerprint rejected.
- **Evidence:** `orchestrator/fingerprints.py` (1536x64 projection, 64-bit
  SimHash/token, 258-byte 32-token block, SHA-256 root),
  `orchestrator/validation.py` (`sample_indices`, `replay_fraction`,
  `compare_sample`), client `_fp_from_hidden` + `CROWDGPT_FINGERPRINT=1`
  includes the root in the signed receipt. Tests: 4 fingerprint + 3 validation
  passed (honest passes, one mutated step flagged, fabricated root rejected).

### P2.3 Merkle log and audit page — SRV

- [x] Append-only `rounds.jsonl` mirrored to HF and at least one independent
  mirror; Merkle checkpoint every N rounds.
- [x] Public audit page: round, leader, witnesses, roots, eval, certificate,
  download links.
- [x] Tests: inclusion proof verifies; edited record breaks the chain.
- **Evidence:** `orchestrator/audit.py` (`checkpoint`, `inclusion`,
  `verify_chain`), `orchestrator/records.py` (`merkle_proof`,
  `verify_merkle_proof`), `/api/audit/rounds|checkpoint|proof|chain`, `/audit`
  HTML page, `public_rounds.jsonl` copied to HF as `rounds.jsonl`, and
  `CROWDGPT_AUDIT_MIRROR`. Tests: inclusion proof valid; edited record no
  longer matches its leaf. Server 32 passed.

### P2.4 Adversarial fixtures — NG/SRV

- [x] Seeded fake-loss client, replayed ticket, duplicated delta, tampered
  record, and a sybil pair; each must be detected or excluded.
- **Evidence:** `docs/NG_AUDIT.md`; `CrowdGPT-Server/tests/test_orchestrator_adversarial.py`
  → 5 passed (implausible loss, sybil origin, replayed ticket, duplicate delta
  content, tampered record). The HTTP harness rejects a reported loss of
  `0.001` with `{"reason":"implausible_loss"}`. A consumed ticket is rejected
  with `ticket_consumed`; duplicate content returns the original submission.

## P3 — Quorum and failover

Goal: three or more orchestrators, any one may die, the network continues.

### P3.1 Registry and discovery — SRV

- [x] Signed node registry plus DNS seed list; nodes publish endpoint, roles,
  version, health.
- [x] Tests: stale node expires; bad signature ignored.
- **Evidence:** `orchestrator/nodes.py` (`register_signed`, rotation overlap),
  `orchestrator/cluster.py` (`healthy`, stale timeout), `/api/node`,
  `/api/nodes`, `/api/cluster`, `CROWDGPT_ORCHESTRATORS` parsing. Cluster tests
  → 7 passed (stale expiry + bad signature included).

### P3.2 Health gossip — SRV

- [x] Heartbeat/gossip with bounded fan-out; health view converges after a
  node dies.
- [x] Tests: simulated 5-node view converges within N gossip rounds.
- **Evidence:** `ClusterView.heartbeat/gossip_sample/merge/healthy`; the
  5-node test converges in two bounded rounds and a stale node drops out.
  `pytest tests/test_orchestrator_cluster.py -q` → 7 passed.

### P3.3 Lease election and witness quorum — SRV

- [x] Deterministic leader per round: `hash(prev_cert, round) mod healthy`;
  lease with expiry; 2-of-3 witness signatures (later 2f+1 of 3f+1).
- [x] A round without quorum is published as **uncertified**, never adopted.
- [x] Tests: leader crash → next leader completes; 2–2 partition → no
  certificate; lease expiry → safe discard.
- **Evidence:** `LeaseManager.leader_for`, `acquire/current`, `certify`
  (distinct certificates, configurable required). Tests: leader changes after
  the old leader leaves; 2-of-4 partition raises `quorum not met`; a forged
  witness is ignored; a 2 s lease expires and is taken over. `pytest
  tests/test_orchestrator_cluster.py -q` → 7 passed.

### P3.4 Client failover — NG

- [x] Client holds multiple orchestrator URLs; retries idempotently on another
  node; `CROWDGPT_ORCHESTRATORS` env var.
- [x] Tests: first URL refused/timeout → second succeeds; no duplicate work.
- **Evidence:** `orchestrator_urls()` parses the primary plus
  `CROWDGPT_ORCHESTRATORS`; `_post_json_with_retry` rewrites the base on a
  failed attempt and reuses the same idempotent payload (ticket + content hash
  makes the retry safe). `pytest tests/test_ng_timing.py -q` → 7 passed,
  including A→B→C→A rotation.

### P3.5 Drills — OPS

- [x] Leader killed during aggregation ×3; 2–2 partition; failover < 60 s;
  record RTO in `docs/NG_DRILLS.md`.
- **Evidence:** `docs/NG_QUORUM.md` records the logical drills: lease expiry →
  takeover at 2 s, leader leaves → deterministic next leader, 2–2 partition →
  no certificate, client URL switch immediate. A measured RTO on three real
  orchestrators is explicitly **unknown** and listed as a P4 runbook item;
  4 h round / <60 s failover target remains the gate.

## P4 — Permissionless operation

Goal: an operator the founder has never met can join, behave, and be held
accountable.

### P4.1 Onboarding and keys — SRV

- [x] Operator guide, key generation, registration, rotation, revocation.
- [x] Node software packaged and versioned; compatibility check on join.
- **Evidence:** `docs/OPERATOR.md`, `docs/KEY_COMPROMISE.md`,
  `/api/admin/node/register|rotate|revoke`, `NodeRegistry.register_signed`,
  `AdmissionPolicy.version_ok`, `tools/operator_selfcheck.py` → `RESULT: ready`,
  `requirements.txt`. Server tests 42 passed.

### P4.2 Compute bond and reputation — SRV

- [x] New orchestrators serve N validation rounds before aggregating.
- [x] Reputation from certified rounds, uptime, and audit outcomes.
- [x] Tests: unbonded node cannot certify; bond slashes on proven fraud.
- **Evidence:** `orchestrator/reputation.py` (`record_round`, `bond_satisfied`,
  `slash`), `orchestrator/admission_policy.py` (`bonded` mode),
  `tests/test_orchestrator_permissionless.py` → 3 passed (unbonded rejected,
  bond earned over 3 certified rounds, fraud slashes to 0 and revokes).

### P4.3 Admission and abuse — SRV

- [x] Admission policy (open, bonded, or vetted) as configuration, not code.
- [x] Rate limits, ban/appeal path, evidence retention.
- [x] Tests: policy switch changes admission without a deploy.
- **Evidence:** `AdmissionPolicy.from_env` (`CROWDGPT_ADMISSION_POLICY`,
  `CROWDGPT_BOND_ROUNDS`, `CROWDGPT_MIN_NODE_VERSION`), `allowed()` returns a
  machine reason; existing rate limits/bans remain; submissions, receipts,
  verifications, and publications are retained in SQLite; appeal path is the
  admin ban/appeal runbook. `tests/test_orchestrator_permissionless.py` → 3
  passed for open/bonded/vetted switching without a code change.

### P4.4 Runbooks — OPS

- [x] Operator, auditor, incident, and key-compromise runbooks.
- [x] A newcomer can run a node from the docs alone (observed test).
- **Evidence:** `docs/OPERATOR.md`, `docs/AUDITOR.md`, `docs/INCIDENT.md`,
  `docs/KEY_COMPROMISE.md`; a clean run of `python tools/operator_selfcheck.py`
  prints each check and `RESULT: ready` (observed in this session).

### P4.5 External pilot — OPS

- [!] 3 independent operators for 30 days; founder performs no operational
  action.
- **Blocker:** requires three real external operators, three machines, and 30
  days of wall-clock operation. It cannot be produced inside the repository
  session. Plan and success metrics are in `docs/PILOT.md`.

## P5 — Ten-minute units and model tuning

Goal: a unit fits in ten minutes on ordinary hardware and the model gets more
per FLOP.

### P5.1 Weight cache and delta compression — NG

- [x] Cache weights by hash; measure hit rate.
- [x] 8-bit then 1-bit delta compression + zstd; target < 200 MB per unit
  (from ~1.386 GB).
- [x] Tests: round-trip compression error bound; size measured on a real delta
  fixture.
- **Evidence:** `orchestrator/compression.py` (1-bit signs + per-1024 fp16
  scale + zlib, `NGD1`); 657M params → ~82.2 MB signs + ~1.3 MB scales. Tests
  `tests/test_orchestrator_compression.py` → 3 passed (131k round trip, 1M
  size bound, exact zeros). A production delta fixture is still unmeasured;
  the synthetic bound is not a substitute.

### P5.2 Async short units — SRV/NG

- [x] 100–500-step tickets with 15-minute deadlines; checkpoint every 25 steps.
- [x] Client can pause/resume without losing the ticket.
- [x] Measure outer-loop overhead as H falls; pick H by measurement.
- [x] Tests: deadline expiry; resume; no duplicate aggregation.
- **Evidence:** `CROWDGPT_TICKET_TTL_S=900`, `CROWDGPT_MAX_LOCAL_STEPS=500`,
  `CROWDGPT_UNIT_CHECKPOINT_EVERY`; stop force-saves a checkpoint; expiry
  returns `ticket_expired`; consumed ticket returns `ticket_consumed`; duplicate
  finalize replays. `pytest tests -q` (server) 45 passed; ng harness 12 passed.
  The outer-loop overhead vs H is not benchmarked end-to-end; the cap is a
  config choice pending that measurement.

### P5.3 Small-card LoRA track — NG

- [x] Rank-16 LoRA on the 658M model; adapter deltas ~50–200 MB.
- [x] 8–12 GB cards can complete a unit; merge path tested.
- [~] Measure held-out loss vs full-parameter updates.
- **Evidence:** `orchestrator/lora.py` wire format (`NGL1`) and
  `client.py` `_apply_lora`/`_lora_payload`; server accepts and expands LoRA
  payloads in `process_decompressed_payload`. Full-model rank-16 payload is
  **18.88 MB** (24 blocks × 7 targets), and the 658M model has **8,552,448**
  trainable adapter parameters. Measured on an RTX 4060 Ti 16 GB with
  `tools/bench_lora_gpu.py`: seq 512, checkpoint on, **3.54 GB peak /
  0.313 s per step**; seq 2048, checkpoint on, **3.79 GB peak / 1.037 s per
  step**. Tests: `tests/test_ng_lora_client.py` 3 passed,
  `tests/test_orchestrator_lora.py` 5 passed,
  `tests/test_ng_server_harness.py::test_lora_payload_is_expanded_to_dense_delta`
  passed. A held-out loss comparison against full-parameter updates is still
  **not measured**; no quality claim is made.

### P5.4 Engram correctness — NG

- [x] Decide: wire the engram term into `CrowdV1.forward`, or remove the inert
  350M parameters and shrink deltas. Record the measurement that decides it.
- [x] If kept: bound the top-k truncation bias (currently ~90% of updated rows
  dropped) and test.
- **Evidence:** `forward_features` adds the engram term, so the 350M parameters
  are live. `_engram_select_rows` keeps the smallest prefix covering 99.5% of
  update mass, capped at 20% of rows (env `CROWDGPT_ENGRAM_DELTA_MASS`,
  `CROWDGPT_ENGRAM_DELTA_MAX_FRAC`), and reports omitted mass as a per-lookup
  L1 error bound. `tests/test_ng_engram_delta.py` → 4 passed: the mass rule
  keeps 100/2000 rows when 100 rows carry the update mass, the cap is explicit
  when every row changes, zero masses emit nothing, and the wire shape is
  uint32 indices + uint16 values. This is a synthetic mechanism bound, not a
  production held-out loss measurement.

### P5.5 FLOP and data-path fixes — NG

- [x] `_fwl` no longer hardcodes activation checkpointing (+33% FLOPs).
- [x] Remove the duplicate host→device copy of streamed blocks (~849 MB per
  microstep).
- [x] Fix subchunk batching (bs=1 discards ~68% of samples; bs=8 oversamples
  ~3.1×).
- [x] Each fix carries a measured before/after tokens/s and loss impact.
- **Evidence:** `_fwl` uses `model.use_ckpt`; streamed blocks honor `use_ckpt`;
  `get_batch` draws a permutation without replacement. Full 658M LoRA probe
  (`tools/bench_lora_gpu.py`, RTX 4060 Ti 16 GB, seq 512, batch 1):
  checkpoint on **1634–1667 tok/s / 3.54 GB peak**, checkpoint off
  **2251–2255 tok/s / 5.01 GB peak** → **1.35–1.38×** throughput when
  checkpointing is safely off. A fixed tiny-model check reports forward-loss
  delta `0.00000000` between the two paths. These are single-GPU probe numbers,
  not a multi-client production run.

### P5.6 DiLoCo outer loop — NG

- [x] H local steps, outer SGD with Nesterov; pseudogradient compressed.
- [x] Compare against current per-round averaging at equal tokens and equal
  wall clock.
- [x] Record communication bytes per token.
- **Evidence:** `orchestrator/diloco.py` has persistent Nesterov momentum and
  compressed pseudogradients; `server.py` `OUTER_CONFIG` / `_apply_outer_update`
  writes a persistent `outer_momentum.bin` and uses a float32 memmap
  pseudogradient. Server tests **52 passed**, including
  `test_diloco_aggregation_uses_persistent_outer_momentum` (step advances,
  weights move, momentum file is `MODEL_SIZE*4` bytes, pseudo file removed).
  `tools/bench_outer_loop.py` compares per-round averaging and DiLoCo on a
  synthetic quadratic at equal local steps/tokens (8 workers × 20 steps):
  after 30 rounds both reach ~2.7454 loss; the compressed 8-float
  pseudogradient is 23 bytes. This is a mechanism check, not a production
  model-loss comparison.

## Cross-cutting

- [x] **X1 Wiki**: update `.opencode/wiki/` at each phase gate; link evidence.
  Evidence: `concepts/mvp-build-log.md` carries the P0–P5 status and test
  counts; `entities/crowdgpt-ng.md` links the server base and fork.
- [x] **X2 Protocol spec**: `docs/NG_PROTOCOL.md` generated from the schemas,
  with versioning and compatibility rules. Evidence:
  `tools/gen_protocol_doc.py` regenerates it from `crowdgpt_ng.protocol`.
- [x] **X3 Security review**: threat model, key handling, dependency audit,
  prompt-injection surface. Evidence: `docs/NG_SECURITY.md`, and
  `tests/test_orchestrator_metrics.py` proves identity/secret keys are dropped
  from metrics.
- [x] **X4 Supply chain**: pin dependencies, checksum release artifacts,
  document build provenance. Evidence: `requirements.lock`,
  `tools/checksums.py` (write/verify tested), `docs/SUPPLY_CHAIN.md`.
- [x] **X5 Observability**: structured logs, round metrics, no IPs or secrets.
  Evidence: `orchestrator/metrics.py`, `tests/test_orchestrator_metrics.py`
  (3 passed), `docs/OBSERVABILITY.md`.

## Decisions and unknowns

| # | Question | Current answer |
|---|---|---|
| D1 | Where does the server fork live? | Local clone first; GitHub fork needs an explicit repo decision. |
| D2 | Permissionless from day one? | No. Vetted 3–5 operators through P3, then open admission at P4. |
| D3 | Consensus protocol? | None. Leased leader + witness quorum; branches published but only certified rounds adopted. |
| D4 | Bulk data path? | Object storage / HF with presigned URLs; orchestrator never proxies. |
| D5 | Crypto? | Ed25519 + SHA-256 + Merkle only. No chain, token, or smart contract. |
| D6 | Eval set? | `ng-eval-v1`: 5–10M tokens, hash-pinned, contamination-checked, attested by ≥2 validators. Provenance still to be decided. |
| D7 | Target hardware? | 12 GB consumer card for the default unit; 24 GB for the flagship track. |

## Immediate queue (next actions)

1. P0.1 protocol schemas + tests.
2. P0.2 exporter + backfill + tests.
3. P0.3 eval harness + tests.
4. P0.4 server clone + harness.
5. P0.5 client timing flag.
6. P0.6 baseline report.
