#!/usr/bin/env python3
"""Short held-out probe: rank-16 LoRA vs full-parameter updates.

Randomly initialised 658M model, a small real text sample from the repository,
equal token budget, and a fixed held-out window set. This measures the update
mechanism on this hardware; it is **not** a production quality comparison and
must not be cited as one. Run:

    python tools/bench_update_tracks.py --steps 30 --seq 128

Measured on an RTX 4060 Ti 16 GB (2026-11-09), 30 steps x 128 tokens:
    LoRA rank 16, AdamW 1e-3 : held-out 12.1004 -> 10.9054, 3.39 GB peak
    full, SGD 1e-4           : held-out 12.1004 -> 12.0952, 6.69 GB peak
The full-parameter arm used a lower learning rate after the 1e-3 run diverged;
both arms saw the same 3,840 training tokens and the same held-out windows.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--eval-windows", type=int, default=4)
    parser.add_argument("--lora-lr", type=float, default=1e-3)
    parser.add_argument("--full-lr", type=float, default=1e-4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(client.TOKENIZER_REPO, trust_remote_code=True)
    text = (ROOT / "README.md").read_text(errors="ignore")
    text += (ROOT / "docs" / "NG_P5_MEASUREMENTS.md").read_text(errors="ignore")
    ids = tokenizer(text)["input_ids"]
    need = (args.steps + args.eval_windows) * (args.seq + 1)
    while len(ids) < need:
        ids = ids * 2
    train = [ids[i * (args.seq + 1):(i + 1) * (args.seq + 1)] for i in range(args.steps)]
    held = [ids[(args.steps + i) * (args.seq + 1):(args.steps + i + 1) * (args.seq + 1)]
            for i in range(args.eval_windows)]

    device = torch.device("cuda")
    client.ENGRAM_ACTIVE = False
    client.LOSS_CHUNK = min(64, args.seq)

    def batch(window):
        t = torch.tensor([window], dtype=torch.long, device=device)
        return t[:, :-1], t[:, 1:]

    def build(lora):
        torch.manual_seed(1234)
        model = client.SotaGPT("CUDA", device, gpu_layers=client.N_LAYERS,
                               bs=1, sl=args.seq)
        model.use_ckpt = True
        if lora:
            client._apply_lora(model, 16, 32.0)
        return model

    def evaluate(model):
        model.eval()
        losses = []
        with torch.no_grad():
            for window in held:
                x, y = batch(window)
                loss, _ = client._fwl(model, x, y, args.seq, True, torch.bfloat16)
                losses.append(float(loss))
        return sum(losses) / len(losses)

    for lora in (True, False):
        model = build(lora)
        trainable = [p for p in model.parameters() if p.requires_grad]
        before = evaluate(model)
        opt = torch.optim.AdamW(trainable, lr=args.lora_lr) if lora else \
            torch.optim.SGD(trainable, lr=args.full_lr)
        start = time.perf_counter()
        model.train()
        for window in train:
            x, y = batch(window)
            opt.zero_grad(set_to_none=True)
            loss, _ = client._fwl(model, x, y, args.seq, True, torch.bfloat16)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        after = evaluate(model)
        print(f"{'lora' if lora else 'full':4s} trainable={sum(p.numel() for p in trainable)} "
              f"before={before:.4f} after={after:.4f} seconds={elapsed:.1f} "
              f"peak_GB={torch.cuda.max_memory_allocated()/1e9:.2f}")
        del model, opt, trainable
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
