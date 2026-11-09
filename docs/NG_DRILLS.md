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
