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
import time
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


def _headers(ip: str, *, mode: str = "dense", rank: int = 0) -> dict[str, str]:
    headers = {
        "X-Forwarded-For": ip,
        "X-Client-Version": "3.2",
        "X-Client-Backend": "CUDA",
        "X-Client-Vram-Gb": "24",
        "X-Client-Gpu-Name": "NVIDIA RTX 4090",
        "X-Client-Batch-Size": "1",
        "X-Client-Autocast": "bf16",
        "X-Client-Optim": "adamw",
    }
    if mode == "lora":
        headers["X-Client-Train-Mode"] = "lora"
        headers["X-Client-Lora-Rank"] = str(rank)
    return headers


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


def _delta_body(task_id: str, params: int, *, loss: float = 2.5, fill: float = 0.0) -> bytes:
    metadata = {
        "taskId": task_id,
        "weightFormat": "bf16",
        "hasEngram": False,
        "engramSparseCount": 0,
        "localSteps": 1,
        "tokensProcessed": 1234,
        "loss": loss,
    }
    raw = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    values = b"\x00\x00" * params
    if fill:
        import torch
        values = torch.full((params,), fill, dtype=torch.bfloat16).view(torch.int16).numpy().tobytes()
    return struct.pack("<I", len(raw)) + raw + values


def _lora_body(task_id: str, *, rank: int = 2, loss: float = 2.4) -> bytes:
    import numpy as np
    from orchestrator.lora import encode_payload

    metadata = {
        "taskId": task_id, "weightFormat": "bf16", "hasEngram": False,
        "engramSparseCount": 0, "localSteps": 1, "tokensProcessed": 1234,
        "loss": loss, "loraRank": rank, "isLoRA": True,
    }
    a = np.full((rank, 4), 0.5, dtype=np.float16)
    b = np.full((4, rank), 0.5, dtype=np.float16)
    payload = encode_payload([{"name": "blocks.0.attn.wq.weight", "offset": 0, "a": a, "b": b}])
    raw = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    return struct.pack("<I", len(raw)) + raw + payload


def test_diloco_aggregation_uses_persistent_outer_momentum(tmp_path, monkeypatch):
    module = _configure(_import_server(_prepare_repo(tmp_path), monkeypatch), monkeypatch)
    client = TestClient(module.app)
    task = _two_client_task(module, client)
    payload = _delta_body(task["taskId"], module.MODEL_SIZE, fill=0.1)
    upload_id = _upload_delta(module, client, task["taskId"], payload)
    final = client.post("/fl/submit/finalize", headers=_headers("10.0.0.1"),
                        json={"uploadId": upload_id})
    assert final.status_code == 200, final.text
    pool = list(module.pools.values())[0]
    path = next(iter(pool["deltas"].values()))

    module.OUTER_CONFIG["OPTIMIZER"] = "diloco"
    module.OUTER_MOMENTUM_FILE = tmp_path / "outer_momentum.bin"
    import torch
    module.global_weights = torch.zeros(module.MODEL_SIZE)
    module.global_step = 0
    module.apply_round_aggregation([{"pool_id": pool["pool_id"], "deltas": [(path, 1.0)]}])

    assert module.global_step == 1
    assert float(module.global_weights[0]) != 0.0
    assert module.OUTER_MOMENTUM_FILE.exists()
    assert module.OUTER_MOMENTUM_FILE.stat().st_size == module.MODEL_SIZE * 4
    assert not (module.WORK_TMP / "outer_pseudograd.f32").exists()


def test_lora_payload_is_expanded_to_dense_delta(tmp_path, monkeypatch):
    module = _configure(_import_server(_prepare_repo(tmp_path), monkeypatch), monkeypatch)
    client = TestClient(module.app)
    task = _two_client_task(module, client, mode="lora", rank=2)
    payload = _lora_body(task["taskId"])
    assert len(payload) < 1024, "adapter payload must be far smaller than a dense delta"
    checksum = hashlib.sha256(payload).hexdigest()
    init = client.post("/fl/submit/init", headers=_headers("10.0.0.1"), json={
        "taskId": task["taskId"], "totalBytes": len(payload), "chunkSize": len(payload),
        "chunks": 1, "checksum": checksum, "weightFormat": "bf16",
        "hasEngram": False, "engramSparseCount": 0, "loss": 2.4,
        "localSteps": 1, "tokensProcessed": 1234, "loraRank": 2, "isLoRA": True,
        "trainedParams": ["adapters"],
    })
    assert init.status_code == 200, init.text
    upload_id = init.json()["uploadId"]
    chunk = client.post("/fl/submit/chunk", headers={
        **_headers("10.0.0.1", mode="lora", rank=2), "Content-Type": "application/octet-stream"},
        params={"uploadId": upload_id, "index": 0}, content=payload)
    assert chunk.status_code == 200, chunk.text
    final = client.post("/fl/submit/finalize", headers=_headers("10.0.0.1"),
                        json={"uploadId": upload_id})
    assert final.status_code == 200, final.text
    assert final.json()["status"] == "success"
    paths = list(module.pools.values())[0]["deltas"]
    assert len(paths) == 1
    stored = Path(next(iter(paths.values())))
    assert stored.stat().st_size >= module.MODEL_SIZE * 2
    # The reconstructed dense stream carries the adapter update at offset 0.
    import torch
    with open(stored, "rb") as f:
        meta_len = struct.unpack("<I", f.read(4))[0]
        f.read(meta_len)
        dense = torch.frombuffer(bytearray(f.read(module.MODEL_SIZE * 2)),
                                 dtype=torch.uint16).view(torch.bfloat16).float()
    assert dense[0].item() > 0


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


def _two_client_task(module, client, *, mode: str = "dense", rank: int = 0):
    first = client.get("/fl/task", headers=_headers("10.0.0.1", mode=mode, rank=rank),
                       params={"skip_weights": "true"})
    assert first.json()["status"] == "wait"
    second = client.get("/fl/task", headers=_headers("10.0.0.2", mode=mode, rank=rank),
                        params={"skip_weights": "true"}).json()
    assert second["taskId"]
    return second


def _upload_delta(module, client, task_id, payload, *, chunk_size=None, order=None):
    checksum = hashlib.sha256(payload).hexdigest()
    chunk_size = chunk_size or len(payload)
    chunks = (len(payload) + chunk_size - 1) // chunk_size
    init = client.post("/fl/submit/init", headers=_headers("10.0.0.1"), json={
        "taskId": task_id, "totalBytes": len(payload), "chunkSize": chunk_size,
        "chunks": chunks, "checksum": checksum, "weightFormat": "bf16",
        "hasEngram": False, "engramSparseCount": 0, "loss": 2.5,
        "localSteps": 1, "tokensProcessed": 1234,
    })
    assert init.status_code == 200, init.text
    upload_id = init.json()["uploadId"]
    for index in (order or range(chunks)):
        part = payload[index * chunk_size:(index + 1) * chunk_size]
        chunk = client.post("/fl/submit/chunk", headers={
            **_headers("10.0.0.1"), "Content-Type": "application/octet-stream"},
            params={"uploadId": upload_id, "index": index}, content=part)
        assert chunk.status_code == 200, chunk.text
    return upload_id


def test_duplicate_finalize_replays_original_result(tmp_path, monkeypatch):
    module = _configure(_import_server(_prepare_repo(tmp_path), monkeypatch), monkeypatch)
    client = TestClient(module.app)
    task = _two_client_task(module, client)
    upload_id = _upload_delta(module, client, task["taskId"],
                              _delta_body(task["taskId"], module.MODEL_SIZE))
    first = client.post("/fl/submit/finalize", headers=_headers("10.0.0.1"),
                        json={"uploadId": upload_id})
    assert first.status_code == 200, first.text
    second = client.post("/fl/submit/finalize", headers=_headers("10.0.0.1"),
                         json={"uploadId": upload_id})
    assert second.status_code == 200, second.text
    assert second.json() == first.json()
    assert len(list(module.pools.values())[0]["deltas"]) == 1
    assert len(module.ORCH.submission.load_submissions(pool_id=list(module.pools)[0])) == 1


def test_out_of_order_chunks_finalize(tmp_path, monkeypatch):
    module = _configure(_import_server(_prepare_repo(tmp_path), monkeypatch), monkeypatch)
    client = TestClient(module.app)
    task = _two_client_task(module, client)
    payload = _delta_body(task["taskId"], module.MODEL_SIZE)
    upload_id = _upload_delta(module, client, task["taskId"], payload,
                              chunk_size=max(1, len(payload) // 3),
                              order=[2, 0, 1])
    final = client.post("/fl/submit/finalize", headers=_headers("10.0.0.1"),
                        json={"uploadId": upload_id})
    assert final.status_code == 200, final.text
    assert final.json()["status"] == "success"


def test_expired_ticket_is_rejected_with_reason(tmp_path, monkeypatch):
    module = _configure(_import_server(_prepare_repo(tmp_path), monkeypatch), monkeypatch)
    client = TestClient(module.app)
    task = _two_client_task(module, client)
    ticket = module.ORCH.admission.get("ip:10.0.0.2")
    assert ticket is not None
    assert ticket.signature and module.ORCH.admission.verify(ticket) is True
    with module.ORCH.store.transaction() as conn:
        conn.execute("UPDATE tickets SET expires_at=? WHERE client_key=?",
                     (time.time() - 100, ticket.client_key))
    payload = _delta_body(task["taskId"], module.MODEL_SIZE)
    init = client.post("/fl/submit/init", headers=_headers("10.0.0.2"), json={
        "taskId": task["taskId"], "totalBytes": len(payload),
        "chunkSize": len(payload), "chunks": 1,
        "checksum": hashlib.sha256(payload).hexdigest(),
        "weightFormat": "bf16", "hasEngram": False, "engramSparseCount": 0,
        "loss": 2.5, "localSteps": 1, "tokensProcessed": 1234,
    })
    assert init.status_code == 403, init.text
    assert init.json()["detail"]["reason"] == "ticket_expired"


def test_finalize_replays_after_restart(tmp_path, monkeypatch):
    repo = _prepare_repo(tmp_path)
    first = _configure(_import_server(repo, monkeypatch), monkeypatch)
    client = TestClient(first.app)
    task = _two_client_task(first, client)
    upload_id = _upload_delta(first, client, task["taskId"],
                              _delta_body(task["taskId"], first.MODEL_SIZE))
    original = client.post("/fl/submit/finalize", headers=_headers("10.0.0.1"),
                           json={"uploadId": upload_id})
    assert original.status_code == 200, original.text

    second = _configure(_import_server(repo, monkeypatch), monkeypatch)
    assert second._restore_orchestrator_state() is True
    replay = TestClient(second.app).post(
        "/fl/submit/finalize", headers=_headers("10.0.0.1"), json={"uploadId": upload_id})
    assert replay.status_code == 200, replay.text
    assert replay.json() == original.json()


def test_presigned_object_upload_and_hash_check(tmp_path, monkeypatch):
    module = _configure(_import_server(_prepare_repo(tmp_path), monkeypatch), monkeypatch)
    client = TestClient(module.app)
    task = _two_client_task(module, client)
    payload = _delta_body(task["taskId"], module.MODEL_SIZE)
    checksum = hashlib.sha256(payload).hexdigest()
    init = client.post("/fl/submit/init", headers=_headers("10.0.0.1"), json={
        "taskId": task["taskId"], "totalBytes": len(payload),
        "chunkSize": len(payload), "chunks": 1, "checksum": checksum,
        "weightFormat": "bf16", "hasEngram": False, "engramSparseCount": 0,
        "loss": 2.5, "localSteps": 1, "tokensProcessed": 1234,
    })
    assert init.status_code == 200, init.text
    body = init.json()
    assert len(init.content) < 2048, "init response must stay control-plane sized"
    grant = body["object"]
    assert grant["method"] == "PUT" and grant["sha256"] == checksum

    # A tampered signature is rejected before any bytes are stored.
    bad = grant["url"].replace("sig=", "sig=0")
    assert client.put(bad, content=payload).status_code == 403
    # An expired grant is rejected.
    expired = grant["url"].replace("exp=", "exp=1")
    assert client.put(expired, content=payload).status_code == 403
    # The signed URL accepts exactly the advertised bytes.
    before = dict(module.CONTROL_PLANE)
    put = client.put(grant["url"], content=payload)
    assert put.status_code == 200, put.text
    assert put.json()["sha256"] == checksum
    assert module.CONTROL_PLANE["requests"] == before["requests"], "object PUT must bypass the control plane"

    final = client.post("/fl/submit/finalize", headers=_headers("10.0.0.1"),
                        json={"uploadId": body["uploadId"], "objectKey": grant["key"],
                              "sha256": checksum})
    assert final.status_code == 200, final.text
    assert final.json()["status"] == "success"
    assert len(list(module.pools.values())[0]["deltas"]) == 1

    # A stored object whose content does not match the expected hash is refused.
    task2 = _two_client_task(module, client)
    payload2 = _delta_body(task2["taskId"], module.MODEL_SIZE)
    checksum2 = hashlib.sha256(payload2).hexdigest()
    init2 = client.post("/fl/submit/init", headers=_headers("10.0.0.1"), json={
        "taskId": task2["taskId"], "totalBytes": len(payload2),
        "chunkSize": len(payload2), "chunks": 1, "checksum": checksum2,
        "weightFormat": "bf16", "hasEngram": False, "engramSparseCount": 0,
        "loss": 2.5, "localSteps": 1, "tokensProcessed": 1234,
    }).json()
    grant2 = init2["object"]
    module.OBJECT_STORE.put(grant2["key"], b"not the delta")
    bad_final = client.post("/fl/submit/finalize", headers=_headers("10.0.0.1"),
                            json={"uploadId": init2["uploadId"], "objectKey": grant2["key"],
                                  "sha256": checksum2})
    assert bad_final.status_code == 400, bad_final.text
    assert bad_final.json()["detail"]["reason"] == "object_hash_mismatch"


def test_closed_round_is_not_resurrected(tmp_path, monkeypatch):
    repo = _prepare_repo(tmp_path)
    first = _configure(_import_server(repo, monkeypatch), monkeypatch)
    client = TestClient(first.app)
    _two_client_task(first, client)
    # Mark the round closed as the watchdog does after aggregation.
    first.ORCH.scheduling.save_round(
        first.round_state["current_round"], state="closed",
        started_at=first.round_state["round_start_time"], duration_hours=4.0,
        last_aggregation_at=time.time(), global_step=first.global_step,
        participating=dict(first.round_state["participating_clients"]))
    second = _configure(_import_server(repo, monkeypatch), monkeypatch)
    assert second._restore_orchestrator_state() is False
    assert second.pools == {}
    assert second.task_tickets == {}


def test_signed_round_record_and_node_identity(tmp_path, monkeypatch):
    module = _configure(_import_server(_prepare_repo(tmp_path), monkeypatch), monkeypatch)
    client = TestClient(module.app)
    _two_client_task(module, client)
    assert module.NODE_ID
    node = client.get("/api/node")
    assert node.status_code == 200
    assert node.json()["nodes"][0]["node_id"] == module.NODE_ID

    module._publish_round_record(module.round_state["current_round"], True, False, 1)
    rec = module.ORCH.publication.get_round(module.round_state["current_round"])["record"]
    assert rec["leader"] == module.NODE_ID
    assert rec["certificate"] and rec["base_weights"] != "0" * 64
    from orchestrator.records import record_hash, verify_certificate
    assert verify_certificate(rec, rec["certificate"][0]) is True
    tampered = dict(rec); tampered["bytes"] = {"down": 0, "up": 0}
    assert verify_certificate(tampered, rec["certificate"][0]) is False
    assert record_hash(rec) != record_hash(tampered)


def test_signed_client_receipt_is_verified(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from orchestrator.identity import NodeIdentity
    from orchestrator.records import canonical_json

    module = _configure(_import_server(_prepare_repo(tmp_path), monkeypatch), monkeypatch)
    client = TestClient(module.app)
    task = _two_client_task(module, client)
    ticket = module.ORCH.admission.get("ip:10.0.0.2")
    key = NodeIdentity.generate()
    receipt = {
        "schema": "ng-1", "ticket_id": ticket.ticket_id, "round": ticket.round,
        "client_id": "device-1",
        "base_weights": hashlib.sha256(f"step:{module.global_step}".encode()).hexdigest(),
        "tokens": 1234, "steps": 1, "loss_samples": ["2.500000"],
        "loss_mean": "2.500000", "fingerprint_root": None,
        "code_version": "3.2", "device": {"backend": "CUDA", "gpu": "RTX 4090"},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(), "nonce": "e" * 64,
    }
    sig = key.sign(canonical_json(receipt).encode())
    headers = {**_headers("10.0.0.2"), "X-Client-Pubkey": key.public_key_hex}
    payload = _delta_body(task["taskId"], module.MODEL_SIZE)
    init = client.post("/fl/submit/init", headers=headers, json={
        "taskId": task["taskId"], "totalBytes": len(payload),
        "chunkSize": len(payload), "chunks": 1,
        "checksum": hashlib.sha256(payload).hexdigest(),
        "weightFormat": "bf16", "hasEngram": False, "engramSparseCount": 0,
        "loss": 2.5, "localSteps": 1, "tokensProcessed": 1234,
        "ticketId": ticket.ticket_id, "receipt": receipt,
        "receiptSig": sig, "clientPubkey": key.public_key_hex,
    })
    assert init.status_code == 200, init.text
    up = init.json()["uploadId"]
    assert client.post("/fl/submit/chunk", headers={**headers, "Content-Type": "application/octet-stream"},
                       params={"uploadId": up, "index": 0}, content=payload).status_code == 200
    final = client.post("/fl/submit/finalize", headers=headers, json={"uploadId": up})
    assert final.status_code == 200, final.text
    assert len(module.ORCH.receipts.all(ticket.round)) == 1


def test_audit_page_and_proof(tmp_path, monkeypatch):
    module = _configure(_import_server(_prepare_repo(tmp_path), monkeypatch), monkeypatch)
    client = TestClient(module.app)
    _two_client_task(module, client)
    module._publish_round_record(module.round_state["current_round"], True, False, 1)
    page = client.get("/audit")
    assert page.status_code == 200 and "CrowdGPT audit" in page.text
    rounds = client.get("/api/audit/rounds").json()["rounds"]
    assert len(rounds) == 1
    rnd = rounds[0]["record"]["round"]
    proof = client.get("/api/audit/proof", params={"round": rnd}).json()
    assert proof["valid"] is True
    assert client.get("/api/audit/chain").json()["ok"] is True


def test_implausible_loss_is_rejected(tmp_path, monkeypatch):
    module = _configure(_import_server(_prepare_repo(tmp_path), monkeypatch), monkeypatch)
    client = TestClient(module.app)
    task = _two_client_task(module, client)
    payload = _delta_body(task["taskId"], module.MODEL_SIZE, loss=0.001)
    checksum = hashlib.sha256(payload).hexdigest()
    init = client.post("/fl/submit/init", headers=_headers("10.0.0.2"), json={
        "taskId": task["taskId"], "totalBytes": len(payload),
        "chunkSize": len(payload), "chunks": 1, "checksum": checksum,
        "weightFormat": "bf16", "hasEngram": False, "engramSparseCount": 0,
        "loss": 0.001, "localSteps": 1, "tokensProcessed": 1234,
    }).json()
    up = init["uploadId"]
    client.post("/fl/submit/chunk", headers={**_headers("10.0.0.2"), "Content-Type": "application/octet-stream"},
                params={"uploadId": up, "index": 0}, content=payload)
    final = client.post("/fl/submit/finalize", headers=_headers("10.0.0.2"), json={"uploadId": up})
    assert final.status_code == 400, final.text
    assert final.json()["detail"]["reason"] == "implausible_loss"


def test_dense_and_lora_clients_form_separate_pools(tmp_path, monkeypatch):
    module = _configure(_import_server(_prepare_repo(tmp_path), monkeypatch), monkeypatch)
    client = TestClient(module.app)
    dense = client.get("/fl/task", headers=_headers("10.0.0.1"),
                       params={"skip_weights": "true"})
    assert dense.json()["status"] == "wait"
    lora = client.get("/fl/task", headers=_headers("10.0.0.2", mode="lora", rank=2),
                      params={"skip_weights": "true"})
    assert lora.json()["status"] == "wait"
    assert len(module.pools) == 2
    assert all(len(pool["clients"]) == 1 for pool in module.pools.values())


def test_lora_submit_to_dense_task_is_rejected(tmp_path, monkeypatch):
    module = _configure(_import_server(_prepare_repo(tmp_path), monkeypatch), monkeypatch)
    client = TestClient(module.app)
    task = _two_client_task(module, client)
    payload = _lora_body(task["taskId"])
    init = client.post("/fl/submit/init", headers=_headers("10.0.0.2"), json={
        "taskId": task["taskId"], "totalBytes": len(payload), "chunkSize": len(payload),
        "chunks": 1, "checksum": hashlib.sha256(payload).hexdigest(),
        "weightFormat": "bf16", "hasEngram": False, "engramSparseCount": 0,
        "loss": 2.4, "localSteps": 1, "tokensProcessed": 1234,
        "loraRank": 2, "isLoRA": True, "trainedParams": ["adapters"],
    })
    assert init.status_code == 200, init.text
    upload_id = init.json()["uploadId"]
    chunk = client.post("/fl/submit/chunk", headers={
        **_headers("10.0.0.2"), "Content-Type": "application/octet-stream"},
        params={"uploadId": upload_id, "index": 0}, content=payload)
    assert chunk.status_code == 200, chunk.text
    final = client.post("/fl/submit/finalize", headers=_headers("10.0.0.2"),
                        json={"uploadId": upload_id})
    assert final.status_code == 400, final.text
    assert final.json()["detail"]["reason"] == "train_mode_mismatch"


def test_lora_plus_engram_finalizes_with_both_updates(tmp_path, monkeypatch):
    import numpy as np
    import torch
    from orchestrator.lora import encode_payload

    module = _configure(_import_server(_prepare_repo(tmp_path), monkeypatch), monkeypatch)
    module.ENG_PARAMS = 16
    module.ENG_NUM_BUCKETS = 4
    module.global_engram_weights = torch.zeros(16)
    client = TestClient(module.app)
    task = _two_client_task(module, client, mode="lora", rank=2)
    assert task["trainMode"] == "lora"
    assert task["loraRank"] == 2
    assert set(task["trainedParams"]) == {"adapters", "engram"}

    a = np.full((2, 4), 0.5, dtype=np.float16)
    b = np.full((4, 2), 0.5, dtype=np.float16)
    adapters = encode_payload([{"name": "blocks.0.attn.wq.weight", "offset": 0, "a": a, "b": b}])
    rows = 2
    tail = (struct.pack("<I", rows)
            + np.array([0, 1], dtype=np.uint32).tobytes()
            + torch.full((rows, 4), 0.5, dtype=torch.bfloat16).view(torch.int16).numpy().tobytes())
    metadata = {
        "taskId": task["taskId"], "weightFormat": "bf16", "hasEngram": True,
        "engramSparseCount": rows, "localSteps": 1, "tokensProcessed": 1234,
        "loss": 2.4, "loraRank": 2, "isLoRA": True,
        "trainedParams": ["adapters", "engram"],
    }
    raw = json.dumps(metadata, separators=(",", ":")).encode()
    payload = struct.pack("<I", len(raw)) + raw + adapters + tail

    init = client.post("/fl/submit/init", headers=_headers("10.0.0.1", mode="lora", rank=2), json={
        "taskId": task["taskId"], "totalBytes": len(payload), "chunkSize": len(payload),
        "chunks": 1, "checksum": hashlib.sha256(payload).hexdigest(),
        "weightFormat": "bf16", "hasEngram": True, "engramSparseCount": rows,
        "loss": 2.4, "localSteps": 1, "tokensProcessed": 1234,
        "loraRank": 2, "isLoRA": True, "trainedParams": ["adapters", "engram"],
    })
    assert init.status_code == 200, init.text
    upload_id = init.json()["uploadId"]
    chunk = client.post("/fl/submit/chunk", headers={
        **_headers("10.0.0.1", mode="lora", rank=2),
        "Content-Type": "application/octet-stream"},
        params={"uploadId": upload_id, "index": 0}, content=payload)
    assert chunk.status_code == 200, chunk.text
    final = client.post("/fl/submit/finalize",
                        headers=_headers("10.0.0.1", mode="lora", rank=2),
                        json={"uploadId": upload_id})
    assert final.status_code == 200, final.text
    assert final.json()["status"] == "success"

    pool = list(module.pools.values())[0]
    stored = Path(next(iter(pool["deltas"].values())))
    with open(stored, "rb") as f:
        meta_len = struct.unpack("<I", f.read(4))[0]
        meta = json.loads(f.read(meta_len))
        assert meta["trainMode"] == "lora"
        assert set(meta["trainedParams"]) == {"adapters", "engram"}
        f.seek(meta_len + 4 + module.MODEL_SIZE * 2)
        assert struct.unpack("<I", f.read(4))[0] == rows


def test_round_close_writes_round_metric(tmp_path, monkeypatch):
    import json
    module = _configure(_import_server(_prepare_repo(tmp_path), monkeypatch), monkeypatch)
    client = TestClient(module.app)
    _two_client_task(module, client)
    module._publish_round_record(module.round_state["current_round"], True, False, 1)
    path = module.DATA_DIR / "round_metrics.jsonl"
    assert path.exists()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["schema"] == "ng-metric-1"
    assert row["round"] == module.round_state["current_round"]
    assert row["accepted"] == 0 and row["deltas"] == 0
    assert row["certified"] is False  # leader-signed only; witness quorum is P6.6
    assert "ip" not in row and "secret" not in row


def test_round_record_uses_evaluator_and_writes_nutrition(tmp_path, monkeypatch):
    import json
    module = _configure(_import_server(_prepare_repo(tmp_path), monkeypatch), monkeypatch)
    client = TestClient(module.app)
    _two_client_task(module, client)
    module._round_eval = lambda held: {
        "mode": "heldout", "loss": "3.250000", "ciLow": "3.100000",
        "ciHigh": "3.400000", "windows": 8, "manifestSha256": "a" * 64,
        "source": "test-double",
    }
    module._publish_round_record(module.round_state["current_round"], True, False, 1)
    rec = module.ORCH.publication.get_round(module.round_state["current_round"])["record"]
    assert rec["eval"]["loss"] == "3.250000" and rec["eval"]["windows"] == 8
    nutrition = json.loads((module.DATA_DIR / "nutrition.json").read_text())
    assert nutrition["schema"] == "ng-nutrition-1"
    assert nutrition["evaluation"]["loss"] == "3.250000"
    assert nutrition["round"] == module.round_state["current_round"]
    assert "ip" not in json.dumps(nutrition).lower() or "limitations" in nutrition


def test_unevaluated_round_stays_explicit(tmp_path, monkeypatch):
    import json
    module = _configure(_import_server(_prepare_repo(tmp_path), monkeypatch), monkeypatch)
    client = TestClient(module.app)
    _two_client_task(module, client)
    module._round_eval = lambda held: None
    module._publish_round_record(module.round_state["current_round"], True, False, 1)
    rec = module.ORCH.publication.get_round(module.round_state["current_round"])["record"]
    assert rec["eval"] is None
    nutrition = json.loads((module.DATA_DIR / "nutrition.json").read_text())
    assert nutrition["evaluation"] == {"status": "unevaluated"}


def test_leaderboard_endpoints_label_provenance(tmp_path, monkeypatch):
    module = _configure(_import_server(_prepare_repo(tmp_path), monkeypatch), monkeypatch)
    client = TestClient(module.app)
    verified = client.get("/api/leaderboard/verified")
    reported = client.get("/api/leaderboard/reported")
    assert verified.status_code == 200 and verified.json()["source"] == "verified"
    assert reported.status_code == 200 and reported.json()["source"] == "reported"


def test_proposal_surface_requires_verified_weight_to_vote(tmp_path, monkeypatch):
    module = _configure(_import_server(_prepare_repo(tmp_path), monkeypatch), monkeypatch)
    client = TestClient(module.app)
    made = client.post("/api/proposals", headers=_headers("10.0.0.1"),
                       json={"title": "next", "body": "more eval"})
    assert made.status_code == 200, made.text
    pid = made.json()["proposalId"]
    listed = client.get("/api/proposals").json()["proposals"]
    assert any(p["proposalId"] == pid for p in listed)
    denied = client.post(f"/api/proposals/{pid}/vote", headers=_headers("10.0.0.1"))
    assert denied.status_code == 403
    assert denied.json()["detail"]["reason"] == "verified_contributor_required"


def test_admin_secret_accepts_header_without_query_string(tmp_path, monkeypatch):
    module = _configure(_import_server(_prepare_repo(tmp_path), monkeypatch), monkeypatch)
    client = TestClient(module.app)
    _two_client_task(module, client)
    r = client.post("/api/admin/force_round_end",
                    headers={**_headers("10.0.0.1"), "X-Admin-Secret": "test-secret"},
                    params={"minutes": 0})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "scheduled"
