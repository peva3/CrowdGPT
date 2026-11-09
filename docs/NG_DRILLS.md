# Restart drills (P1.5)

## SIGKILL durability drill

`CrowdGPT-Server/tests/test_restart_drill.py` forks a real process and sends
`SIGKILL` at three boundaries:

| Boundary | Assertion after reopen |
|---|---|
| after a committed round + pool + ticket | round 7, both pool members, and the ticket are present |
| inside an open SQLite transaction | the uncommitted round 9 is gone (WAL rollback) |
| while a half-written object exists | the upload row is still `open`, no object is committed |

Run: `cd CrowdGPT-Server && pytest tests/test_restart_drill.py -q` → 3 passed.

## Server restart / exactly-once

`Crowdgpt-ng/tests/test_ng_server_harness.py` simulates a process restart with a
fresh module namespace over the same data directory:

- round, pools, members, tasks, and live tickets are reconstructed;
- a finalized upload replays its original result after restart;
- a closed round is never resurrected (`_restore_orchestrator_state()` returns
  false), so an interrupted aggregation ends as an explicit discard, not a
  half-applied round;
- the pool contains exactly one delta after a duplicate finalize.

Run: `cd Crowdgpt-ng && pytest tests/test_ng_server_harness.py -q` → 8 passed.

## Known boundary

The server test drives the app in-process; it does not spawn uvicorn and kill
it. The SIGKILL drill covers the durable store and object commit boundaries,
which are the parts that can corrupt state. A production drill should repeat
the same sequence against a running uvicorn process and is listed as an
operator runbook item in P4.

## P6.0 on-machine cluster simulation

`CrowdGPT-Server/tools/sim_cluster.py` launches real node processes with
separate SQLite stores, separate Ed25519 keys, and separate HTTP ports. This is
a **single-machine simulation**. It is not a production multi-node RTO and does
not satisfy the P3 or P4 operational gates.

Run:

```bash
cd CrowdGPT-Server
python tools/sim_cluster.py --root /tmp/ng-sim --port 7320 --out /tmp/ng-sim/report.json
```

Configuration: 3 nodes, quorum 2, lease TTL 3 s. Observed on 2026-11-09:

| Drill | Result |
|---|---|
| Signed registry convergence | all 3 nodes saw all 3 signed records |
| Leader killed ×3 mid-round | takeover each time; RTO 3.021 s, 3.020 s, 3.053 s |
| Duplicate finalize to two nodes | first `created=true`, second `false`, third `false`; each node count = 1 |
| Restart under write load | 40 submissions written; 41 before restart, 41 after, 41 after replay; no loss, no duplicates |
| 2–2 partition, 4 nodes, global quorum 3 | neither side acquired a lease (`neither_certified=true`) |
| 2-of-3 witness certificates across processes | 2 certificates accepted; 1 certificate rejected with `quorum not met: 1/2` |

The takeover RTO is dominated by the configured lease TTL, not by election
work, so it is a lower-bound shape check rather than a tuned number. Node death
while a **client** waits is covered separately by the client failover tests
(P6.4), not by this runner.

Evidence artifact: `/tmp/ng-sim/report.json` from the run above.

## P7 packaging and peer-routing readiness

### Aggregation memory floor

`CrowdGPT-Server/tools/bench_aggregation_memory.py` allocates the real
657,787,392-param base and 350,000,640-param engram as float32 masters, then
applies one 10M-parameter delta chunk.

Observed on this machine (2026-11-09):

| Measure | Value |
|---|---|
| Master bytes | 4,031,152,128 (4.03 GB) |
| Allocation time | 0.236 s |
| Peak RSS after allocation | 4.12 GB |
| Apply time for all chunks | 0.619 s |
| Peak RSS after apply | 4.16 GB |

The old 2 vCPU / 4 GB operator minimum is therefore too low. Use at least
8 GB RAM for aggregation; 16 GB is comfortable. This probe does not include
file streaming, merge buffers, or the HTTP server's own footprint.

### Static-peer routing and handoff

`tests/test_orchestrator_peer_service.py` → 6 passed:

- A-issued signed ticket and task replicate to B; B can consume the ticket.
- Replaying the same ticket does not create a second row or reset consumption.
- A conflicting local ticket is not overwritten by a peer.
- Reused nonces and stale timestamps are rejected.
- Submission content hashes are recorded once.
- A lease requires the peer acknowledgement and the registry announcement reaches
  the peer.

`Crowdgpt-ng/tests/test_ng_server_harness.py` exercises the production
`/cluster/*` routes in-process: ticket/task, round snapshot, submission,
signed registry, witness signature, and `/operator` page.

### Multi-process drill

`CrowdGPT-Server/tools/sim_cluster.py` was rerun after the P7 changes. It is
still a **single-machine simulation**, not a production RTO:

| Drill | Result |
|---|---|
| Registry convergence | 3/3 |
| Leader killed ×3 | takeover each time; RTO 3.022 / 3.021 / 3.056 s |
| Duplicate finalize | exactly once, counts [1,1,1] |
| Restart under 40 writes | 41 before, 41 after, 41 after replay; no loss/duplicates |
| 2–2 partition, global quorum 3 | neither side certified |
| 2-of-3 witness certificates | 2 accepted; 1 rejected `quorum not met: 1/2` |

The production `server.py` peer routes and the lab `sim_node.py` are separate
processes; the full training path is not run across three 658M-model nodes on
this machine. The simulation proves the control-plane mechanics only.
