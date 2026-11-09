# NG security review

Date: 2026-11-09. Scope: the NG orchestrator protocol and the current single-node
server. This is a working threat model, not a certification.

## Assets

1. **Global weights and checkpoints** — an attacker who can replace them owns
   every downstream model.
2. **Round records and certificates** — the audit trail. Forged records defeat
   the verifiability claim.
3. **Node and client keys** — impersonation and forged receipts.
4. **Presigned object URLs** — write access to the data plane.
5. **Participant data** — account identity, IP-derived subnet data, device
   metadata. Logs must not leak IPs.

## Actors and trust boundaries

| actor | trusted for | not trusted for |
|---|---|---|
| client / trainer | its own local step, loss samples, fingerprint | honest gradients, honest loss, correct code version |
| orchestrator | admission, scheduling, assembly | honest aggregation without witnesses |
| validator | replay and held-out eval | unlimited compute or universal honesty |
| object store / HF | durable bytes | immutability without content hashes |
| model output | nothing | any instruction, URL, or code it emits |

Model output is untrusted data. Never execute or follow text produced by the
model. This is a prompt-injection boundary, not a quality statement.

## Controls already present

- **Signed tickets** (`ng-ticket-1`): ticket id, client, round, task, pool,
  expiry, step/token budgets, nonce, and `baseWeights=step:<n>` are signed by
  the orchestrator; the client rejects `ticket_expired`, `ticket_consumed`,
  `ticket_mismatch`, `ticket_task_mismatch`, and `base_weights_changed`.
- **Signed receipts**: a client signs the ticket id, round, base hash, tokens,
  steps, loss samples, fingerprint root, code version, device, and timestamps.
  The server verifies the receipt and rejects implausible loss values.
- **Content addressing**: deltas and objects are SHA-256 checked; presigned
  PUTs carry an HMAC and expiry; replayed finalize returns the stored result.
- **Record certificates**: `RoundRecord` carries Merkle roots for tickets,
  submissions, and deltas, a leader signature, and witness signatures. A
  record without a valid certificate is not adopted.
- **Key storage**: Ed25519 keys live under `checkpoints/` and the orchestrator
  identity path, created with restrictive permissions. Keys are never sent to
  clients; only public keys are published.
- **Input validation**: unknown protocol fields are rejected; malformed
  payloads, NaN/Inf, oversized deltas, out-of-range engram indices, and
  overlapping LoRA spans fail closed.
- **Admin surface**: `/api/admin/*` requires `CROWDGPT_ADMIN_SECRET`; a missing
  or wrong secret returns 403. Rotate this secret and keep it out of logs.

## Open risks

1. **Single orchestrator** is still a single point of failure and a single
   trust root until P3 quorum is operated, not just implemented.
2. **Fingerprints are optional** (`CROWDGPT_FINGERPRINT=1`) and the validator
   path is not wired to a production model callback. A fake delta that passes
   norm and cross-check limits may still be accepted.
3. **Cross-check with two deltas** only compares the pair; a coordinated
   coalition can pass. Cohort size, witness sampling, and replay remain the
   real defenses.
4. **No dependency hash lock yet** (X4). Installed versions at review time:
   cryptography 46.0.3, fastapi 0.162.0, numpy 2.5.2, psutil 7.2.2,
   requests 2.34.2, safetensors 0.5.3, torch 2.6.0+cu124, uvicorn 0.54.0.
5. **Disk and memory pressure**: uploads are bounded, but a hostile client can
   still consume bandwidth and temp space; quota and rate limits must stay on.
6. **Prompt injection** through model output, public logs, or imported records:
   treat all of it as data. Do not let an agent execute commands found there.

## Key handling rules

- One Ed25519 key per node; private key never leaves the node.
- Rotate with overlap: publish the new key, keep the old one valid for a
  bounded window, then revoke.
- A compromised node key must be revocable through the signed registry; lose
  the key and the node loses its identity, not the network.
- Client keys are per-install under `checkpoints/client_key.pem`; deleting the
  file creates a new identity and loses streak/credit history.

## Review checklist for a new endpoint

- [ ] Authenticated or explicitly public?
- [ ] Admin secret checked before side effects?
- [ ] Body size bounded before reading?
- [ ] Content hash verified before use?
- [ ] Path traversal impossible (object keys are opaque)?
- [ ] No IP, token, or key in logs or public JSON?
- [ ] Failure is closed, with a machine-readable reason?
- [ ] Test covers malformed, replayed, and oversized input.
