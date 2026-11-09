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

- [ ] `crowdgpt_ng/eval.py`: hash-pinned token shards, fixed window stride,
  mean negative log-likelihood in nats, bootstrap confidence interval.
- [ ] Manifest records dataset hash, token count, model weights hash, code
  commit, and tokenizer revision; refuse to run if a hash does not match.
- [ ] CPU-feasible sampling mode (N windows, seeded) for the 2-core
  coordinator.
- [ ] Tests: known logits produce the expected loss; manifest mismatch
  refuses; sampling is deterministic.
- **Evidence:** `pytest tests/test_ng_eval.py -q`

### P0.4 Server test harness and baseline — SRV

- [ ] Fork/clone the server repository; record the base commit in
  `docs/NG_SERVER_BASE.md`.
- [ ] Add an offline pytest harness that imports `server.py` against a
  temporary data directory.
- [ ] Cover `/fl/task`, `/fl/round_status`, `/fl/submit/init`,
  `/fl/submit/chunk`, `/fl/submit/finalize`, and one admin round-end path.
- [ ] Record baseline: bytes per orchestrated round, requests per round,
  in-memory state that a restart loses.
- [ ] Tests: restart drops in-flight tickets (documented, then fixed in P1).
- **Evidence:** `pytest` output + `docs/NG_SERVER_BASE.md`

### P0.5 Client timing instrumentation — NG

- [ ] Behind `CROWDGPT_TIMING_LOG=<path>`, append one JSONL row per phase:
  queue wait, weight fetch, training, delta upload, total; include bytes and
  token count.
- [ ] No behavior change when the variable is unset; no secrets or IPs.
- [ ] Tests: unset → no file; set → valid JSONL; failure paths still emit a
  row with `ok=false`.
- **Evidence:** `pytest tests/test_ng_timing.py -q`

### P0.6 Baseline report — OPS

- [ ] `docs/NG_BASELINE.md`: p50/p95 unit wall clock, bytes down/up, tokens
  per unit, reported vs held-out loss with the metric's provenance.
- [ ] State what is **unknown** (per-client timing is not published today).
- **Evidence:** report committed; numbers traceable to `data/rounds.jsonl`

## P1 — Stateless single orchestrator

Goal: one orchestrator that can be killed at any moment without losing or
duplicating work, and that never proxies bulk bytes.

### P1.1 Server structure — SRV

- [ ] Split admission/scheduling, submission/aggregation, verification, and
  publication into modules with explicit interfaces.
- [ ] Replace in-memory rounds/pools/tickets with SQLite WAL (Postgres later
  only if measured necessary).
- [ ] Persist tickets, round state, and pool membership; reconstruct on boot.
- [ ] Tests: every module can be imported and exercised without the HTTP layer.

### P1.2 Durable tickets and idempotent submission — SRV

- [ ] Signed nonce tickets bound to `(round, base_weights, client, step_budget,
  token_budget, expires_at)`.
- [ ] `submit/init|chunk|finalize` idempotent by ticket + content hash; retries
  return the original result, never a second delta.
- [ ] Expired tickets rejected with a machine-readable reason.
- [ ] Tests: duplicate finalize, out-of-order chunks, replay after restart,
  expired ticket.

### P1.3 Presigned data plane — SRV/NG

- [ ] Upload URLs issued by the orchestrator, bytes go to object storage (or
  HF), SHA-256 verified on finalize.
- [ ] Client verifies downloaded weights against `base_weights` before
  training.
- [ ] Orchestration request/response bytes measured per round.
- [ ] Tests: bad hash rejected; expired URL rejected; orchestrator sees no bulk
  payload.

### P1.4 Client cache, resume, prefetch — NG

- [ ] Content-addressed weight cache (`sha256` directory), reused across
  rounds; hit rate logged.
- [ ] Resume a paused/interrupted unit from the last checkpoint instead of
  refetching weights.
- [ ] Prefetch the next ticket's weights while uploading the current delta.
- [ ] Tests: second run hits cache; interrupted run resumes; no duplicate
  upload.

### P1.5 Restart drill — SRV

- [ ] `kill -9` during aggregation and during upload; restart; the round
  completes or is explicitly discarded, never half-applied.
- [ ] Test asserts exactly one accepted delta per ticket and no lost tickets.
- **Evidence:** drill log in `docs/NG_DRILLS.md`

## P2 — Signed records and validators

Goal: a round is a signed object that anyone can audit; fake work is detected
and excluded.

### P2.1 Node identity and signed records — SRV/NG

- [ ] Ed25519 node keys; `NodeRecord` published at registration and rotation.
- [ ] Leader signs `RoundRecord`; witnesses co-sign; certificate embedded.
- [ ] Client `StepReceipt` signed with the account key (or session key).
- [ ] Tests: tampered field rejected; unknown key rejected; rotated key obeys
  an overlap window.

### P2.2 Activation fingerprints and replay — NG/SRV

- [ ] Fingerprint sampled activations (LSH; ~258 bytes per 32 tokens) into
  `StepReceipt.fingerprint_root`.
- [ ] Validator mode recomputes a random sample of steps and compares
  fingerprints; flags divergence.
- [ ] Replay a random fraction of submissions end-to-end; sampling alone is
  not sufficient for a single fake.
- [ ] Tests: honest client passes; one mutated step flagged; fabricated
  fingerprint rejected.

### P2.3 Merkle log and audit page — SRV

- [ ] Append-only `rounds.jsonl` mirrored to HF and at least one independent
  mirror; Merkle checkpoint every N rounds.
- [ ] Public audit page: round, leader, witnesses, roots, eval, certificate,
  download links.
- [ ] Tests: inclusion proof verifies; edited record breaks the chain.

### P2.4 Adversarial fixtures — NG/SRV

- [ ] Seeded fake-loss client, replayed ticket, duplicated delta, tampered
  record, and a sybil pair; each must be detected or excluded.
- **Evidence:** `docs/NG_ADVERSARIAL.md` with test output.

## P3 — Quorum and failover

Goal: three or more orchestrators, any one may die, the network continues.

### P3.1 Registry and discovery — SRV

- [ ] Signed node registry plus DNS seed list; nodes publish endpoint, roles,
  version, health.
- [ ] Tests: stale node expires; bad signature ignored.

### P3.2 Health gossip — SRV

- [ ] Heartbeat/gossip with bounded fan-out; health view converges after a
  node dies.
- [ ] Tests: simulated 5-node view converges within N gossip rounds.

### P3.3 Lease election and witness quorum — SRV

- [ ] Deterministic leader per round: `hash(prev_cert, round) mod healthy`;
  lease with expiry; 2-of-3 witness signatures (later 2f+1 of 3f+1).
- [ ] A round without quorum is published as **uncertified**, never adopted.
- [ ] Tests: leader crash → next leader completes; 2–2 partition → no
  certificate; lease expiry → safe discard.

### P3.4 Client failover — NG

- [ ] Client holds multiple orchestrator URLs; retries idempotently on another
  node; `CROWDGPT_ORCHESTRATORS` env var.
- [ ] Tests: first URL refused/timeout → second succeeds; no duplicate work.

### P3.5 Drills — OPS

- [ ] Leader killed during aggregation ×3; 2–2 partition; failover < 60 s;
  record RTO in `docs/NG_DRILLS.md`.

## P4 — Permissionless operation

Goal: an operator the founder has never met can join, behave, and be held
accountable.

### P4.1 Onboarding and keys — SRV

- [ ] Operator guide, key generation, registration, rotation, revocation.
- [ ] Node software packaged and versioned; compatibility check on join.

### P4.2 Compute bond and reputation — SRV

- [ ] New orchestrators serve N validation rounds before aggregating.
- [ ] Reputation from certified rounds, uptime, and audit outcomes.
- [ ] Tests: unbonded node cannot certify; bond slashes on proven fraud.

### P4.3 Admission and abuse — SRV

- [ ] Admission policy (open, bonded, or vetted) as configuration, not code.
- [ ] Rate limits, ban/appeal path, evidence retention.
- [ ] Tests: policy switch changes admission without a deploy.

### P4.4 Runbooks — OPS

- [ ] Operator, auditor, incident, and key-compromise runbooks.
- [ ] A newcomer can run a node from the docs alone (observed test).

### P4.5 External pilot — OPS

- [ ] 3 independent operators for 30 days; founder performs no operational
  action.
- **Evidence:** uptime report + certified round count.

## P5 — Ten-minute units and model tuning

Goal: a unit fits in ten minutes on ordinary hardware and the model gets more
per FLOP.

### P5.1 Weight cache and delta compression — NG

- [ ] Cache weights by hash; measure hit rate.
- [ ] 8-bit then 1-bit delta compression + zstd; target < 200 MB per unit
  (from ~1.386 GB).
- [ ] Tests: round-trip compression error bound; size measured on a real delta
  fixture.

### P5.2 Async short units — SRV/NG

- [ ] 100–500-step tickets with 15-minute deadlines; checkpoint every 25 steps.
- [ ] Client can pause/resume without losing the ticket.
- [ ] Measure outer-loop overhead as H falls; pick H by measurement.
- [ ] Tests: deadline expiry; resume; no duplicate aggregation.

### P5.3 Small-card LoRA track — NG

- [ ] Rank-16 LoRA on the 658M model; adapter deltas ~50–200 MB.
- [ ] 8–12 GB cards can complete a unit; merge path tested.
- [ ] Measure held-out loss vs full-parameter updates.

### P5.4 Engram correctness — NG

- [ ] Decide: wire the engram term into `CrowdV1.forward`, or remove the inert
  350M parameters and shrink deltas. Record the measurement that decides it.
- [ ] If kept: bound the top-k truncation bias (currently ~90% of updated rows
  dropped) and test.
- **Evidence:** before/after held-out loss and delta size.

### P5.5 FLOP and data-path fixes — NG

- [ ] `_fwl` no longer hardcodes activation checkpointing (+33% FLOPs).
- [ ] Remove the duplicate host→device copy of streamed blocks (~849 MB per
  microstep).
- [ ] Fix subchunk batching (bs=1 discards ~68% of samples; bs=8 oversamples
  ~3.1×).
- [ ] Each fix carries a measured before/after tokens/s and loss impact.

### P5.6 DiLoCo outer loop — NG

- [ ] H local steps, outer SGD with Nesterov; pseudogradient compressed.
- [ ] Compare against current per-round averaging at equal tokens and equal
  wall clock.
- [ ] Record communication bytes per token.

## Cross-cutting

- [ ] **X1 Wiki**: update `.opencode/wiki/` at each phase gate; link evidence.
- [ ] **X2 Protocol spec**: `docs/NG_PROTOCOL.md` generated from the schemas,
  with versioning and compatibility rules.
- [ ] **X3 Security review**: threat model, key handling, dependency audit,
  prompt-injection surface (model outputs are untrusted data).
- [ ] **X4 Supply chain**: pin dependencies, checksum release artifacts,
  document build provenance.
- [ ] **X5 Observability**: structured logs, round metrics, no IPs or secrets.

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
