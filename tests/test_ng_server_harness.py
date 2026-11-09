"""Offline harness for the unmodified CrowdGPT coordinator.

The public server repository does not ship `krum_stream` or `slowapi`, so this
module imports `server.py` against small local stand-ins and a temporary data
directory. The tests pin the wire behaviour we must preserve while the
orchestrator is made replaceable (P0.4 / P1): a pooled task, a chunked delta
upload, the admin round-end seam, and reconstruction of pools, tasks, and live
tickets after a restart.

Run with:
    CROWDGPT_SERVER_DIR=/path/to/CrowdGPT-Server pytest tests/test_ng_server_harness.py -q
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import os
import shutil
import struct
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

NG_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = Path(
    os.environ.get("CROWDGPT_SERVER_DIR", str(NG_ROOT.parent / "CrowdGPT-Server"))
).resolve()
STUB_DIR = Path(__file__).resolve().parent / "server_stubs"

pytestmark = pytest.mark.skipif(
    not (SERVER_DIR / "server.py").exists(),
    reason=f"coordinator checkout not found at {SERVER_DIR}",
)

_counter = itertools.count(1)


def _headers(ip: str) -> dict[str, str]:
    return {
        "X-Forwarded-For": ip,
        "X-Client-Version": "3.2",
        "X-Client-Backend": "CUDA",
        "X-Client-Vram-Gb": "24",
        "X-Client-Gpu-Name": "NVIDIA RTX 4090",
        "X-Client-Batch-Size": "1",
        "X-Client-Autocast": "bf16",
        "X-Client-Optim": "adamw",
    }


def _prepare_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "coordinator"
    repo.mkdir()
    shutil.copy2(SERVER_DIR / "server.py", repo / "server.py")
    shutil.copy2(SERVER_DIR / "public_log.py", repo / "public_log.py")
    orchestrator = SERVER_DIR / "orchestrator"
    if orchestrator.is_dir():
        shutil.copytree(orchestrator, repo / "orchestrator")
    return repo


def _import_server(repo: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(STUB_DIR))
    monkeypatch.syspath_prepend(str(repo))
    # public_log keeps module-level state (its output directory); reload it for
    # every simulated process so a "restart" cannot inherit the first PUB.
    sys.modules.pop("public_log", None)
    name = f"ng_coordinator_{next(_counter)}"
    spec = importlib.util.spec_from_file_location(name, repo / "server.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure(module, monkeypatch: pytest.MonkeyPatch):
    import torch

    monkeypatch.setenv("CROWDGPT_ADMIN_SECRET", "test-secret")
    module.ADMIN_SECRET = "test-secret"

    # A 64-parameter base model and no engram keeps CPU/disk cost trivial while
    # exercising the exact payload and upload code paths.
    module.MODEL_SIZE = 64
    module.ENG_PARAMS = 0
    module.ENG_NUM_BUCKETS = 4
    module.MODEL_CONFIG["dim"] = 4
    module.MODEL_CONFIG["vocabSize"] = 8
    module.MATCHMAKING["POOL_SIZE"] = 2
    module.HARDEN["MIN_TRAIN_TIME_S"] = 0
    module.HARDEN["MIN_SUBMIT_INTERVAL_S"] = 0
    module.HARDEN["MIN_FREE_DISK_TASK_BYTES"] = 0
    module.HARDEN["TRUSTED_PROXY_NETS"] = []
    module._min_client_version = ""
    module._block_unversioned = False

    module.global_weights = torch.zeros(module.MODEL_SIZE)
    module.global_engram_weights = torch.zeros(0)
    module.dataset_info.update(
        {
            "initialized": True,
            "totalChunks": 1,
            "validChunkIndices": [0],
            "subChunkBytes": 10 * 1024 * 1024,
        }
    )
    module.data_position.update(
        {
            "currentChunkIdx": 0,
            "epoch": 0,
            "permutation": [0],
            "permutationPos": 0,
            "consumed_through_pos": -1,
        }
    )
    module.round_state["current_round"] = 1
    module.round_state["last_aggregation_time"] = 0
    module.round_state["is_aggregating"] = False
    module.round_state["force_end_at"] = None
    return module


def _delta_body(task_id: str, params: int) -> bytes:
    metadata = {
        "taskId": task_id,
        "weightFormat": "bf16",
        "hasEngram": False,
        "engramSparseCount": 0,
        "localSteps": 1,
        "tokensProcessed": 1234,
        "loss": 2.5,
    }
    raw = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    return struct.pack("<I", len(raw)) + raw + (b"\x00\x00" * params)


def test_pooled_round_flow_and_request_count(tmp_path, monkeypatch):
    module = _configure(_import_server(_prepare_repo(tmp_path), monkeypatch), monkeypatch)
    client = TestClient(module.app)
    requests: list[str] = []

    def call(method: str, path: str, ip: str, **kwargs):
        requests.append(f"{method} {path}")
        return client.request(method, path, headers=_headers(ip), **kwargs)

    status = call("GET", "/fl/round_status", "10.0.0.1")
    assert status.status_code == 200

    first = call("GET", "/fl/task", "10.0.0.1", params={"skip_weights": "true"})
    assert first.status_code == 200
    assert first.json()["status"] == "wait"
    assert first.json()["poolFill"] == 1

    second = call("GET", "/fl/task", "10.0.0.2", params={"skip_weights": "true"})
    assert second.status_code == 200
    task = second.json()
    assert task["taskId"]
    assert task["modelSize"] == 64
    assert task["crossCheck"]["enabled"] is True

    # The first client is re-matched into the now-active pool and gets the same task.
    third = call("GET", "/fl/task", "10.0.0.1", params={"skip_weights": "true"})
    assert third.status_code == 200
    assert third.json()["taskId"] == task["taskId"]

    payload = _delta_body(task["taskId"], module.MODEL_SIZE)
    checksum = hashlib.sha256(payload).hexdigest()
    init = call(
        "POST",
        "/fl/submit/init",
        "10.0.0.1",
        json={
            "taskId": task["taskId"],
            "totalBytes": len(payload),
            "chunkSize": len(payload),
            "chunks": 1,
            "checksum": checksum,
            "weightFormat": "bf16",
            "hasEngram": False,
            "engramSparseCount": 0,
            "loss": 2.5,
            "localSteps": 1,
            "tokensProcessed": 1234,
        },
    )
    assert init.status_code == 200, init.text
    upload_id = init.json()["uploadId"]

    requests.append("POST /fl/submit/chunk")
    chunk = client.post(
        "/fl/submit/chunk",
        headers={**_headers("10.0.0.1"), "Content-Type": "application/octet-stream"},
        params={"uploadId": upload_id, "index": 0},
        content=payload,
    )
    assert chunk.status_code == 200, chunk.text
    assert chunk.json()["receivedChunks"] == 1

    final = call("POST", "/fl/submit/finalize", "10.0.0.1", json={"uploadId": upload_id})
    assert final.status_code == 200, final.text
    assert final.json()["status"] == "success"
    assert final.json()["loss"] == 2.5

    pools = list(module.pools.values())
    assert len(pools) == 1
    assert len(pools[0]["clients"]) == 2
    assert len(pools[0]["deltas"]) == 1
    assert module.db["tasks"]
    # Submitting consumes the client's ticket; the pooled peer still holds one.
    assert len(module.task_tickets) == 1

    forced = call(
        "POST",
        "/api/admin/force_round_end",
        "10.0.0.1",
        params={"secret": "test-secret", "minutes": 0},
    )
    assert forced.status_code == 200, forced.text
    assert forced.json()["status"] == "scheduled"
    assert module.round_state["force_end_at"] is not None

    # Baseline for the docs: one client's partial round is 8 HTTP requests.
    assert len(requests) == 8, requests


def test_restart_reconstructs_round_pools_and_tickets(tmp_path, monkeypatch):
    repo = _prepare_repo(tmp_path)
    first = _configure(_import_server(repo, monkeypatch), monkeypatch)
    client = TestClient(first.app)

    assert client.get(
        "/fl/task", headers=_headers("10.0.0.1"), params={"skip_weights": "true"}
    ).json()["status"] == "wait"
    assert client.get(
        "/fl/task", headers=_headers("10.0.0.2"), params={"skip_weights": "true"}
    ).status_code == 200

    assert first.task_tickets, "first process should hold tickets"
    assert first.db["tasks"], "first process should hold tasks"
    assert first.pools, "first process should hold pools"

    # Simulated process restart: same files on disk, a fresh module namespace.
    second = _configure(_import_server(repo, monkeypatch), monkeypatch)
    assert second.ORCH is not None, "orchestrator package should be copied by the harness"
    assert second._restore_orchestrator_state() is True

    second_client = TestClient(second.app)
    status = second_client.get("/fl/round_status", headers=_headers("10.0.0.3"))
    assert status.status_code == 200

    assert second.task_tickets, "live tickets should survive a restart"
    assert second.db["tasks"], "tasks should be rebuilt from persisted pools"
    assert len(second.pools) == 1
    assert second.round_state["current_round"] == first.round_state["current_round"]
    assert len(next(iter(second.pools.values()))["clients"]) == 2
    assert second.client_pool_map, "pool membership should be reconstructed"
