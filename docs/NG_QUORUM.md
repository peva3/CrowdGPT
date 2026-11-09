# Registry, gossip, leases, and quorum (P3)

## Registry and discovery

`orchestrator/nodes.py` stores `NodeRecord`s. `register_signed()` accepts a
record only when the record's own Ed25519 key signed its canonical JSON; a bad
signature is ignored. `resolve(at)` enforces `valid_from`/`valid_until`;
`rotate(overlap_s)` leaves the old key valid during the overlap. `/api/node`,
`/api/nodes`, and `/api/cluster` expose the live view.

`CROWDGPT_ORCHESTRATORS` is the client seed list; `orchestrator_urls()` returns
the primary plus the comma-separated alternatives, and
`_post_json_with_retry()` rotates to the next base after a failed attempt.

## Health gossip

`ClusterView` keeps `node_health(node_id, generation, last_seen, endpoint)`.
`heartbeat()` updates one node; `gossip_sample(k, seed)` picks a bounded
fan-out; `merge()` applies a peer view; `healthy(timeout_s)` drops stale nodes.
The 5-node test converges in two gossip rounds and a 1000-second-old node
expires.

## Leases and quorum

`LeaseManager.leader_for(round, prev_cert, healthy)` is deterministic:
`sha256(prev_cert:round) mod sorted(healthy)`. `acquire()` grants a TTL lease;
a live lease blocks another leader; an expired lease can be taken over.

`certify(record, signatures, required=N)` verifies distinct certificates and
raises `quorum not met` below N. A 2-of-4 partition therefore certifies
nothing; a forged witness signature is ignored.

## Client failover

`CROWDGPT_ORCHESTRATORS` env var: `https://a,https://b,https://c`.
`_post_json_with_retry` rewrites the base URL on a failed attempt and retries
the same idempotent request. Tests assert the A→B→C→A rotation.

## Evidence

`CrowdGPT-Server/tests/test_orchestrator_cluster.py` → 7 passed:
5-node convergence, stale expiry, leader changes after crash, lease
acquire/expiry/takeover, 2-2 partition without a certificate, bad witness
signature ignored, bad node signature ignored.

`Crowdgpt-ng/tests/test_ng_timing.py` → 7 passed, including URL failover.

## RTO status

The lease-expiry takeover is 2 s in the unit test; the URL switch is immediate.
A measured RTO on three real orchestrators is **unknown** until the P4
deployment; the target is under one round (4 h) and the failover URL test
covers the client side. This is recorded as a P4 runbook item rather than
claimed as measured.
