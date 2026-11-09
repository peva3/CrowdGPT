"""The client protocol module and the server record module must agree byte-for-byte.

They are separate implementations so the server can be deployed without the
client package. This test pins the shared contract: canonical JSON, SHA-256
digests, sorted-leaf Merkle roots, and Ed25519 signatures.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT.parent / "CrowdGPT-Server"
if not (SERVER / "orchestrator" / "records.py").exists():
    pytest.skip("server checkout not present", allow_module_level=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SERVER))

from crowdgpt_ng import protocol as ng  # noqa: E402
from orchestrator import records as srv  # noqa: E402


def test_canonical_json_and_digest_match():
    payload = {"b": 2, "a": {"z": 3, "y": [1, 2]}, "n": None, "u": "é"}
    assert ng.canonical_bytes(payload) == srv.canonical_json(payload).encode("utf-8")
    assert ng.digest_of(payload) == srv.digest_of(payload)


def test_merkle_root_matches_sorted_duplicate_last():
    leaves = [hashlib.sha256(x).hexdigest() for x in (b"a", b"b", b"c", b"d", b"e")]
    assert ng.merkle_root(leaves) == srv.merkle_root(leaves)
    assert ng.merkle_root([]) == srv.merkle_root([])
    assert ng.merkle_root(list(reversed(leaves))) == srv.merkle_root(leaves)


def test_signature_is_verifiable_by_the_other_module():
    payload = {"schema": "ng-1", "round": 7, "value": "x"}
    private_hex, public_hex = ng.generate_keypair()
    signer = "node-" + public_hex[:16]
    envelope = ng.sign_payload(payload, private_hex, signer)
    assert srv.verify_client_receipt(payload, envelope["pubkey"], envelope["sig"]) is True

    from orchestrator.identity import NodeIdentity
    key = NodeIdentity.generate()
    sig = key.sign(srv.canonical_json(payload).encode("utf-8"))
    back = ng.verify_payload({"payload": payload, "alg": "ed25519",
                              "signer": signer, "pubkey": key.public_key_hex,
                              "sig": sig})
    assert back == payload


def test_client_verifies_server_round_certificate():
    from orchestrator.identity import NodeIdentity
    from orchestrator.records import ZERO_ROOT, build_round_record, sign_record

    key = NodeIdentity.generate()
    record = build_round_record(
        round_no=5, opened_at="2026-11-09T00:00:00Z", closed_at="2026-11-09T01:00:00Z",
        base_weights="a" * 64, aggregate_weights="b" * 64,
        leader="node-" + key.public_key_hex[:16], witnesses=(),
        tickets_root="1" * 64, submissions_root="2" * 64, deltas_root="3" * 64,
        prev_record_hash=ZERO_ROOT, eval=None, bytes_down=1, bytes_up=2,
        code_versions=("3.2",))
    signed = sign_record(record, key, record["leader"])
    assert ng.verify_round_certificate(signed, signed["certificate"][0]) is True
    tampered = dict(signed); tampered["bytes"] = {"down": 9, "up": 9}
    assert ng.verify_round_certificate(tampered, signed["certificate"][0]) is False
