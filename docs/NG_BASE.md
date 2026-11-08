# CrowdGPT-ng base

This repository is the MVP fork of CrowdGPT used to test large structural
changes before they are proposed upstream. It is deliberately separate from the
maintenance checkout at `../CrowdGPT/` so bug fixes can continue there without
disturbing experiments here.

## What "base" means

`main` is the state we expect upstream to reach when the founder reviews and
merges the open PRs. It is not upstream `main` plus unfinished work-in-progress:
it is `Vxtzq/CrowdGPT@main` plus every open PR except the draft removal.

- Upstream base: `b7d5b39` (`Merge pull request #83 from
  peva3/fix/finding-7-installer-hardening`)
- Integration branch: `main`
- Base commit: `a83cd59` (`chore: ignore local opencode metadata`)

## Merged PRs

Merged with `--no-ff` so each PR keeps its own commit and merge record.

| PR | Branch | Merge commit | Topic |
|----|--------|--------------|-------|
| #85 | `fix/cohort-preflight-placement` | `8103f7f` | Cohort placement VRAM headroom preflight |
| #86 | `fix/head-offload-cache` | `debe445` | Prefer a GPU head over head offload |
| #76 | `fix/finding-1-upload-abort` | `18e1e64` | Keep background uploads alive across pause/stop |
| #80 | `fix/finding-2-adaptive-checkpointing` | `a33a728` | VRAM-aware activation checkpointing |
| #81 | `fix/finding-3-streamed-block-copies` | `30d6606` | Avoid duplicate streamed-block parameter copies |
| #78 | `fix/finding-4-oom-optimizer-moments` | `9927f87` | Preserve optimizer moments on loss-chunk OOM rungs |
| #79 | `fix/finding-5-weight-fetch-buffer` | `532cedc` | Single buffer for inline weight fetch |
| #82 | `fix/finding-6-backend-restart` | `bb2b099` | Restart training after GPU/backend change |
| #77 | `fix/finding-8-stream-timeouts` | `44fed65` | Bound stalled weight-download reads |
| #83 | `fix/finding-7-installer-hardening` | in upstream base | Installer hardening |

## Excluded: PR #84 (dead HiveMind modules)

`chore/finding-9-remove-dead-hivemind` is **not** part of this base. The PR is a
draft whose own `docs/DEAD_CODE_HIVEMIND.md` says: "Do not merge without an
explicit decision from the maintainer." Deleting 985 lines of reference code
before that decision would remove a possible starting point for the distributed
loss-check/orchestrator work.

The branch's second commit (`7708c63`, `.gitignore` entries for `.opencode/`)
is included on its own; the merge was reverted in `af8c7a5` and the hygiene
commit cherry-picked as `a83cd59`.

## Conflicts resolved during integration

Both conflicts were in `SotaGPT.__init__` in `client.py`, where every PR adds a
line after `head_on_gpu`:

- **#80 vs #86** — kept #86's `force_head_cpu` guard and added #80's
  `_u8`/`use_ckpt` decision.
- **#81 vs #80** — kept #80's `use_ckpt` block and #81's
  `cache_streamed_params` check, sharing one `_u8` computation instead of
  defining it twice.

No semantic changes beyond combining the two intents.

## Verification

Run from a scratch directory whose `CrowdGPT` entry points at this checkout
(the workspace suite hardcodes `CrowdGPT/`):

```bash
rm -rf /tmp/ng-verify && mkdir -p /tmp/ng-verify
cp -r ../tests /tmp/ng-verify/tests
ln -s /app/crowdgpt/Crowdgpt-ng /tmp/ng-verify/CrowdGPT
cd /tmp/ng-verify && PYTHONDONTWRITEBYTECODE=1 /app/crowdgpt/.venv/bin/python -m pytest tests -q
```

Result at base `a83cd59`: **171 passed, 3 skipped**.
The maintenance checkout `../CrowdGPT/` reports 159 passed, 15 skipped.
The difference is the new features enabled by the merged PRs.

`python -m compileall` on `client.py`/`installer.py`/`tools` and `bash -n
install.sh` also pass.

`tools/sign_release.py verify` fails with "client.py/client.html changed after
signing" — expected, because these PRs intentionally do not re-sign the release.
The workspace signature test converts that state into a skip.

## Remotes

```
base      /app/crowdgpt/CrowdGPT      (read-only source of PR branches)
upstream  https://github.com/Vxtzq/CrowdGPT.git
```

There is no writable `origin` yet. Creating a GitHub repository is a GitHub-side
action that needs explicit approval; until then the fork lives locally. When a
remote exists, push `main` and use `ng/*` branches for MVP work.

## Working rules for this repo

- `main` stays the reviewed base. Do features on `ng/<topic>` branches.
- Rebase on `upstream/main` after the founder merges; then drop the
  corresponding `--no-ff` merges rather than keeping duplicate history.
- Never reintroduce the workspace test suite inside this repo; it lives at
  `../tests/`.
- Experiments that change observable behavior need a measurement, not just a
  passing test. See `../../PROJECT.md`.
