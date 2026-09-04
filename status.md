# GridFM → vLLM serving — full working status (recovery doc)

_Last updated: 2026-09-04. Purpose: survive a local crash/HD failure. Contains
**no secrets** (see “Secrets” section)._

## Objective

Enable GridFM to be served on vLLM (`/pooling` endpoint, PowerFlow
reconstruction), wire it into **algorithm-nexus**, and publish trained
checkpoints to the Hugging Face Hub.

## Repositories, branches, local paths

- **algorithm-nexus**: `/home/romeokienzler/gitco/algorithm-nexus`
  - PR branch: `feat/gridfm-graphkit-vllm-serving` → **PR IBM/algorithm-nexus#220 (DRAFT)**.
  - Remotes: `fork` = git@github.com:romeokienzler/algorithm-nexus.git (writable);
    `origin` = https://github.com/IBM/algorithm-nexus.git (upstream).
- **gridfm-graphkit**: `/home/romeokienzler/gitco/gridfm-graphkit`
  - Remote `origin` = git@github.com:gridfm/gridfm-graphkit.git.
  - Serving branch: **`feat/vllm-serving`** — PUSHED to origin, tip
    **`f4c60b7`** (`chore(release): bump version to 0.9.1`) on top of
    **`9141f95`** (`feat(vllm): serve GridFM PowerFlow reconstruction via vLLM /pooling`),
    which sits on `origin/main` `6be23bd`.
  - Rebased worktree used for the push: `/tmp/gridfm-vllm-wt` (branch `feat/vllm-serving`).
  - Main repo working tree is on branch `feat/docker-image` with UNCOMMITTED user
    changes (examples/config/HGNS_OPF_datakit_case14.yaml,
    examples/config/HGNS_PF_datakit_case14.yaml, uv.lock) — DO NOT DISTURB.

## What is DONE

1. **Serving code** implemented on `feat/vllm-serving`: `gridfm_graphkit/vllm/`
   subpackage (custom `GridFMGNS` pooling model, `gridfm_pf_reconstruction` IO
   processor, `export.py`, `graph_codec.py`, config/types/utils, plugins) +
   `[vllm]` extra (`vllm>=0.26,<0.27`, safetensors, timm) + entry points
   (`vllm.general_plugins` → `gridfm_gns`; `vllm.io_processor_plugins` →
   `gridfm_pf_reconstruction`). Tests: `tests/test_vllm_graph_codec.py`,
   `tests/test_vllm_pipeline.py`.
2. **Rebase onto latest main** done (only `pyproject.toml` conflicted — resolved
   to keep BOTH main’s `[tool.uv]` find-links block and the serving `vllm` extra).
   Version bumped **0.9.0 → 0.9.1**.
3. **Branch pushed** to `gridfm/gridfm-graphkit` (`feat/vllm-serving`, tip `f4c60b7`).
4. **HF models published (public):**
   - `gridfm/gridfm-pf-reconstruction` — case14_ieee, 2000 scen, 20 ep,
     GNS_heterogeneous 1.3M params; normalizer baseMVA 85.996. Published by user
     `mangalisom`.
   - `romeokienzler/gridfm-pf-case57-tiny` — case57_ieee, hidden_size=12, seed 0;
     commit `8239575`; files README.md/config.json/model.safetensors (416
     tensors, 5.16 MB). Published by user `romeokienzler`.
5. **e2e proven earlier** on the bluevela H100: case14 POST `/pooling` → HTTP 200,
   correct shapes (result.json: num_buses 14, num_gens 5, emb_dim 32).

## IN-FLIGHT / INTERRUPTED step — integration tests on vela

- **Not yet run against the rebased 0.9.1 source.** The `rsync` of the rebased
  source to vela **FAILED due to DNS** ("Temporary failure in name resolution"
  for `login3.bluevela.rmf.ibm.com") — a transient network issue, needs retry.
- Driver staged **locally** at `/tmp/run_all.sh` (runs `gpu_e2e.sh` then
  `pytest tests/test_vllm_graph_codec.py tests/test_vllm_pipeline.py`,
  prints `INTEGRATION_RESULT: PASS/FAIL`). Must be rsync’d to
  `vela:/u/rkie/gridfm-vllm-e2e/run_all.sh`.
- Resume recipe:
  1. `rsync -az --exclude '.git' --exclude '__pycache__' --exclude '*.egg-info' --exclude 'venv' /tmp/gridfm-vllm-wt/ rkie@login3.bluevela.rmf.ibm.com:/u/rkie/gridfm-vllm-e2e/gridfm-graphkit/`
  2. `scp /tmp/run_all.sh rkie@login3...:/u/rkie/gridfm-vllm-e2e/run_all.sh`
  3. `bsub -q normal -n 4 -gpu "num=1" -G grp_partnership_mat -o run_all.out -e run_all.err bash /u/rkie/gridfm-vllm-e2e/run_all.sh`
  4. Wait server-side (avoid rapid ssh polling → fail2ban); read `run_all.out`.

### Vela harness facts

- Host: `ssh rkie@login3.bluevela.rmf.ibm.com` (H100 nodes via LSF `bsub`).
- Dir `/u/rkie/gridfm-vllm-e2e`: `gpu_e2e.sh` (full serve+client e2e, idempotent
  env setup), `client_test.py`, `export_dummy.py`, `venv` (vllm 0.26.0, torch
  2.11.0+cu130, CUDA avail), `model_dir`, `gridfm-graphkit` (editable install,
  **plain copy, not a git repo** — refresh via rsync; GitHub is NOT reachable
  from vela: `git@github.com` → publickey denied).
- gpu_e2e.sh serve cmd: `vllm serve <model_dir> --runner pooling
  --trust-remote-code --skip-tokenizer-init --enforce-eager --dtype float32
  --io-processor-plugin gridfm_pf_reconstruction --enable-mm-embeds --port 8765`.
- **fail2ban**: rapid repeated ssh polling gets the KEY rejected (not a timeout),
  and each retry RESETS the ban; recovery needs a long zero-attempt quiet window.

## CCC (case57 export) facts

- Host `ssh ccc-login4.pok.ibm.com` (separate from vela; `/dccstor` GPFS).
- Inputs: `/dccstor/gridfm/genco_results/pf_3seeds/models/case57_tiny_seed1.pt`
  (state dict, `model.` prefixed, 416 tensors, hidden_size 12) and
  `/dccstor/gridfm/genco_results/pf_3seeds/normalizer_stats/case57_tiny_seed1.pt`
  (nested `{case57_ieee: {baseMVA_orig:100.0, baseMVA:111.83, vn_kv_max:1.0}}`
  → unwrap to flat dict for export).
- Config: case57 PF config from gridfm-graphkit `main`
  (`examples/config/HGNS_PF_datakit_case57.yaml`), `hidden_size` 48→**12**,
  `seed: 0`. Tiny config staged at `vela? no →` CCC `/u/rkie/case57_tiny_pf.yaml`.
- torch env on CCC: `/u/rkie/venvs/venv_gridfm-graphkit/bin/python` (torch 2.8 +
  safetensors + huggingface_hub).
- Exported model dir on CCC: `/u/rkie/gridfm-pf-case57-tiny-export`
  (config.json `architectures:["GridFMGNS"]`, `num_classes:0`,
  `pretrained_cfg.{gridfm_config,normalizer_stats}` + model.safetensors + README).
- Publish script (no secret): `/u/rkie/hf_publish_romeo.py` (whoami guard,
  create_repo public, upload_folder).

## REMAINING steps (user’s ordered plan)

1. **Run integration tests on vela** (see resume recipe) — verify PASS.
2. **Create PR** for `feat/vllm-serving` on `gridfm/gridfm-graphkit`; request
   review from **`albanpuech`** (collaborator; GH handle confirmed); **inform
   Alban**. (No PR exists yet.)
3. **Publish gridfm-graphkit 0.9.1 to PyPI** — from `feat/vllm-serving`.
   ⚠ NEEDS a PyPI token/release rights on the `gridfm` project (not yet
   available). 0.9.0 is already burned on PyPI and contains **no** vLLM code.
4. **algorithm-nexus wiring finalize (PR #220):** regenerate `uv.lock` +
   `requirements-ecosystem.txt` + `requirements-candidate.txt` (uv can resolve
   once 0.9.1 is on PyPI), run `uv run nexus validate package
   packages/gridfm-graphkit`, then mark PR #220 **ready for review**.

## BLOCKERS

- **PyPI token** for the gridfm-graphkit 0.9.1 release (step 3) — required before
  step 4 (uv.lock currently committed with `SKIP=uv-lock` because 0.9.1 isn’t
  resolvable yet).
- Serving code targets `vllm>=0.26,<0.27`; algorithm-nexus `product` extra is on
  `vllm>=0.28` → `gridfm-graphkit[vllm]` intentionally added to **ecosystem +
  candidate only**, NOT product, until validated on 0.28 (user decision).

## algorithm-nexus edits already in PR #220 (commit 04105db)

- `pyproject.toml`: ecosystem `gridfm-graphkit==0.9.1` (bare); candidate
  `gridfm-graphkit[vllm]==0.9.1`; `[tool.uv.extra-build-dependencies]
  torch-scatter = ["torch"]` shim (must sit after all flat `[tool.uv]` keys).
- `packages/gridfm-graphkit/models/gridfm-pf-reconstruction/{model.yaml,usage.md}`.

## Secrets (NOT stored in this file — by design)

- Three HF tokens were pasted in chat during this work (read-only “download”;
  `mangalisom` write; `romeokienzler` write). **None are written to any tracked
  file.** The `romeokienzler` token was used via env var and `shred -u`’d on CCC.
- **Recommendation: rotate/revoke every token that was pasted into chat**, since
  chat transcripts retain them. No PyPI token has been provided or stored.
