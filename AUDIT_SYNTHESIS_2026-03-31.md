# Full Audit Synthesis — CLIP-HBA-Mem

**Date:** 2026-03-31
**Agents Run:** @architecture-auditor, @ml-correctness-auditor, @pipeline-integrity-checker, @reproducibility-agent, @slurm-reviewer, @io-and-storage-auditor, @code-quality-inspector, @experiment-tracking-reviewer, @meg-mlp-head-auditor, @meg-preprocessing-auditor, @meg-data-integrity-auditor, @meg-vs-clip-alignment-auditor

---

## CRITICAL Issues (Must Fix)

### C1. `classnames66` First-Character Bug — 3 Files

**Agents:** @meg-vs-clip-alignment-auditor, @architecture-auditor

`[x[0] for x in classnames66]` extracts the **first character** of each string (e.g., `'metallic; artificial'` -> `'m'`), not the full label. The group MEG checkpoint was **trained with full strings** (`train_meg_things_pipeline.py:871`), so inference with single characters produces wrong text embeddings.

| File | Line | Current (WRONG) | Fix |
|---|---|---|---|
| `inference_meg_group_pipeline.py` | 386 | `[x[0] for x in classnames66]` | `classnames66` |
| `train_meg_individual_pipeline.py` | 356 | `[x[0] for x in classnames66]` | `classnames66` |
| `inference_meg_individual_pipeline.py` | 144 | `[x[0] for x in classnames66]` | `classnames66` |

**Note:** `extract_embeddings_meg.py:121` and `train_meg_things_pipeline.py:871` already use the correct full strings. `tokenized_prompts` is NOT in the state dict — it's recomputed at init from classnames, so this bug silently corrupts inference.

---

### C2. `n_tps` NameError in `extract_embeddings_meg.py`

**Agents:** @code-quality-inspector, @meg-vs-clip-alignment-auditor

`n_tps` and `tp_desc` are defined only in the `combined_lamem_memcat` branch (line 515-516) but referenced unconditionally at line 521. Running with `--training_data lamem` crashes:

```
NameError: name 'n_tps' is not defined
```

**Fix:** Move `n_tps = len(range(...))` above the `if/else` block so both branches define it.

---

### C3. `strict=False` Checkpoint Load in Individual Training

**Agent:** @meg-vs-clip-alignment-auditor (prior audit context)

| File | Line | `map_location` | `strict` | Issue |
|---|---|---|---|---|
| `extract_embeddings_meg.py:178-180` | 178 | `'cpu'` | `True` (explicit) | **Correct** |
| `inference_meg_group_pipeline.py:420-423` | 420 | **Missing** | Implicit (True) | Missing `map_location` |
| `inference_meg_individual_pipeline.py:178-181` | 178 | **Missing** | Implicit (True) | Missing `map_location` |
| `train_meg_individual_pipeline.py:449-452` | 452 | **Missing** | **`strict=False`** | Intentional for partial text encoder load, but risky |

`strict=False` silently ignores missing/extra keys. Acceptable for the partial text-encoder load but should be documented with a comment explaining why.

---

### C4. `clip_lora_model` Missing `train()` Override

**Agent:** @ml-correctness-auditor

`CLIPHBAMem` and `CLIPFrozenMLP` override `train()` to force the backbone into eval mode. `clip_lora_model` does **not** — the frozen vision encoder may enter training mode (affecting BatchNorm/Dropout behavior).

**Fix:** Add `train()` override to `clip_lora_model` in `train_mem_pipeline.py`.

---

### C5. Missing `/tmp` Cache Cleanup in All SLURM Scripts

**Agent:** @slurm-reviewer

All 3 SLURM scripts (`extract_embeddings.slurm`, `train_clip_mlp.slurm`, `train_mem.slurm`) cache datasets in `/tmp/lamem_cache` and `/tmp/memcat_cache` but never clean up. Preempted/failed jobs leave 10+ GB behind.

**Fix:** Add `trap cleanup EXIT` at top of each script.

---

### C6. Incomplete Fold Loops in SLURM Scripts

**Agent:** @slurm-reviewer

- `train_clip_mlp.slurm:77`: `for FOLD in 1; do` — only fold 1 (comment says "all 5")
- `train_mem.slurm:93`: `for FOLD in 2 3; do` — only folds 2-3

**Fix:** Parameterize via `FOLD_START`/`FOLD_END` env vars or `seq 1 $N_FOLDS`.

---

### C7. Checkpoints Save Only `model.state_dict()` — Missing Metadata

**Agents:** @io-and-storage-auditor, @experiment-tracking-reviewer

`train_mem_pipeline.py:544` saves only `model.state_dict()`. CLAUDE.md requires: epoch, optimizer state, config, val metric, timestamp.

**Fix:** Save a dict: `{'model_state': ..., 'epoch': ..., 'optimizer_state': ..., 'config': ..., 'val_loss': ..., 'val_rho': ..., 'git_hash': ...}`.

---

## WARNING Issues

| # | Issue | Agent(s) | Location |
|---|---|---|---|
| W1 | Missing `map_location='cpu'` in `torch.load()` | @architecture-auditor | `inference_meg_group_pipeline.py:420`, `inference_meg_individual_pipeline.py:178` |
| W2 | DoRA application order reversed (vision->text vs text->vision everywhere else) | Prior audit | `train_meg_individual_pipeline.py:427-438` |
| W3 | No checkpoint resumption after SLURM preemption | @slurm-reviewer | All training scripts |
| W4 | Hardcoded absolute path in SLURM scripts (`/panfs/.../jenkm22/...`) | @slurm-reviewer | All 3 `.slurm` files |
| W5 | `run_mem_training()` returns only `best_rho` but callers expect `(best_val_loss, best_rho)` | @experiment-tracking-reviewer | `train_mem_pipeline.py:795` |
| W6 | No W&B / MLflow / experiment tracking integration | @experiment-tracking-reviewer, @reproducibility-agent | Codebase-wide |
| W7 | Race condition: multiple SLURM jobs append to same `sweep_results.csv` without file locking | @io-and-storage-auditor | `train_mem_hyperparam_search.py:384` |
| W8 | `fix_pipeline.py` is dead code (patch already applied) | @architecture-auditor, @code-quality-inspector | `CLIP-HBA/fix_pipeline.py` |
| W9 | `train_perceptclip_pipeline.py` is orphaned (never imported) | @architecture-auditor | `functions/train_perceptclip_pipeline.py` |
| W10 | No wall time / epoch duration logged in training history CSVs | @experiment-tracking-reviewer | `train_mem_pipeline.py:478-535` |
| W11 | Git hash not captured in single-run training (only in sweeps) | @experiment-tracking-reviewer | `train_mem.py` |
| W12 | `EmbeddingDataset` docstring says 768-dim but also serves 66-dim MEG embeddings | @pipeline-integrity-checker | `train_mem_pipeline.py:188-194` |
| W13 | Broad `except Exception:` in hyperparameter search scripts | @code-quality-inspector | Multiple files |
| W14 | Inconsistent `pathlib.Path` vs string f-string path construction | @io-and-storage-auditor | `train_mem_pipeline.py` |
| W15 | `tensorboard` in requirements.txt but never imported/used | @experiment-tracking-reviewer | `requirements.txt:81-82` |

---

## SUGGESTION Issues

| # | Issue | Agent |
|---|---|---|
| S1 | Extract MemCat subpath lookup into shared utility (duplicated in MemDataset + PerceptCLIPDataset) | @code-quality-inspector |
| S2 | Decompose `_run_mem_training_impl` (~200 lines) into smaller functions | @architecture-auditor |
| S3 | Migrate `print()` to `logging` module (215 print statements across 17 files) | @code-quality-inspector |
| S4 | Add shape assertions in MEG extraction (`pred_emb_3d.shape == (n_tp, B, 66)`) | @pipeline-integrity-checker |
| S5 | Extract normalization constants to shared module (`CLIP_HBA_NORM_MEAN`, etc.) | @meg-preprocessing-auditor |
| S6 | Add metadata to extracted `.pt` files (checkpoint path, DoRA config, timestamp) | @pipeline-integrity-checker |
| S7 | Add environment.yml alongside requirements.txt for conda reproducibility | @reproducibility-agent |
| S8 | Document `train_window_size=0` rationale inline | @meg-vs-clip-alignment-auditor |
| S9 | Add `input_dim` validation in `MLPOnlyHead` (assert embedding dim matches config) | @pipeline-integrity-checker |

---

## Clean Bills of Health

| Audit | Status |
|---|---|
| **@meg-preprocessing-auditor** | All PASS — CLIP-HBA normalization consistent across all MEG datasets; PerceptCLIPDataset correctly isolated |
| **@meg-mlp-head-auditor** | All PASS — `input_dim=66` everywhere in MEG path; no output activation; no `input_dim=768` in MEG training |
| **@meg-data-integrity-auditor** | All PASS — model_type strings consistent; fold CSV naming correct; 1-indexed throughout; `.pt` naming matches downstream expectations |
| **@ml-correctness-auditor** (core invariants) | Backbone freezing correct (CLIPHBAMem, CLIPFrozenMLP); loss function correct; no label leakage; folds pre-defined |
| **@reproducibility-agent** (seeding) | All seeds set comprehensively; DataLoader worker seeding correct; cudnn.deterministic=True |

---

## Local Output Directory Sizes (Reference)

| Directory | Size |
|---|---|
| `sweep_meg_out/` | 129 MB |
| `sweep_out/` | 41 MB |
| `preds/` | 206 MB |
| `models/` | empty |
