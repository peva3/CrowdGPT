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

The takeover RTO is dominated by the configured lease TTL, not by election
work, so it is a lower-bound shape check rather than a tuned number. Node death
while a **client** waits is covered separately by the client failover tests
(P6.4), not by this runner.

Evidence artifact: `/tmp/ng-sim/report.json` from the run above.
