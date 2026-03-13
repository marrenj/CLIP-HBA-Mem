# CLIP-HBA-Mem — Full Audit Report

**Date:** 2026-03-13
**Agents:** @architecture-auditor · @ml-correctness-auditor · @pipeline-integrity-checker · @reproducibility-agent · @slurm-reviewer · @io-and-storage-auditor · @code-quality-inspector · @experiment-tracking-reviewer
**Files Covered:** ~40 Python files, 1 SLURM script, configs, docs

---

## CRITICAL (Fix Before Any Publication or Cluster Run)

### C1. Incomplete Checkpoint Metadata
**Agents:** io-auditor, architecture-auditor, reproducibility-agent
**File:** `CLIP-HBA/functions/train_mem_pipeline.py:369` (and all other training pipelines)

Checkpoints save **only** `model.state_dict()` — missing epoch, optimizer state, config dict, val metric, and timestamp. Cannot resume training on preemption; cannot reproduce exact checkpoint conditions. CLAUDE.md explicitly requires all five fields.

**Fix:**
```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config,
    'val_rho': best_rho,
    'val_loss': best_val_loss,
}, checkpoint_path)
```

---

### C2. Image Normalization Mismatch
**Agent:** pipeline-integrity-checker
**File:** `CLIP-HBA/functions/train_mem_pipeline.py:42–47`

`MemDataset` uses custom mean `[0.530, 0.481, 0.419]` / std `[0.276, 0.266, 0.282]` while `PerceptCLIPDataset` uses standard CLIP `[0.481, 0.458, 0.408]` / `[0.269, 0.261, 0.276]`. The frozen CLIP-HBA backbone expects whichever normalization it was trained with — feeding mismatched statistics silently corrupts every feature vector.

**Fix:** Verify CLIP-HBA backbone's original training normalization and enforce it uniformly across all dataset classes; document in a constants file.

---

### C3. Code Duplication of Core Classes
**Agents:** architecture-auditor, code-quality-inspector

Four pipeline files each independently define:
- `seed_everything()` — 4 copies
- `DoRALayer` — 4 copies
- `apply_dora_to_ViT()` — 4 copies
- Dataset classes — multiple near-identical variants

Any bug fix or change must be applied to all 4+ locations, and they already diverge (e.g., DoRA bias freezing inconsistency).

**Fix:** Create `CLIP-HBA/utils/` with `seeds.py`, `dora.py`, `datasets.py`; replace all duplicates with shared imports.

---

### C4. No Automated Cross-Validation Aggregation
**Agent:** experiment-tracking-reviewer

`clip_hba_mem_results.md` has empty `[X]` placeholders for folds 2–5. No script aggregates per-fold CSVs into mean ± std. Reported metrics cannot be reconstructed from code alone.

**Fix:** Create `aggregate_cv_results.py` that reads all fold history CSVs and outputs a mean ± std summary table.

---

### C5. Behavior Training Val/Test Splits Non-Reproducible
**Agent:** reproducibility-agent
**File:** `CLIP-HBA/functions/train_behavior_things_pipeline.py:504`

`random_split(dataset, [train_size, test_size])` is called **without a seeded generator**. `seed_everything()` is called first but `random_split` uses its own internal state unless a `Generator` is passed.

**Fix:**
```python
g = torch.Generator().manual_seed(config['random_seed'])
train_set, test_set = random_split(dataset, [train_size, test_size], generator=g)
```

---

### C6. No Package Version Pinning / No environment.yml
**Agents:** reproducibility-agent, experiment-tracking-reviewer
**File:** `requirements.txt`

`torch`, `numpy`, `scipy`, `scikit-learn`, `transformers` are all unpinned. Only `clip` is locked (to a git commit hash — correct).

**Fix:** Pin all packages to exact versions; add `environment.yml` with Python 3.11 and CUDA 11.8 channel.

---

### C7. No Experiment Tracking (W&B / git hash)
**Agents:** reproducibility-agent, experiment-tracking-reviewer

Zero `wandb` integration anywhere in the codebase. No git commit hash is logged in configs, checkpoints, or CSV logs. Sweep configs save `config.json` but regular training does not.

**Fix (minimum viable):** Log `git rev-parse HEAD`, run timestamp, and random seed into every `config.json` and checkpoint dict; add `wandb` integration for dashboard visibility.

---

### C8. SLURM Scripts Missing from Designated Directory
**Agents:** slurm-reviewer, architecture-auditor

CLAUDE.md specifies `slurm/` should contain all job scripts. The only script (`train_mem.slurm`) lives in `CLIP-HBA/`, not `slurm/`. No eval, preprocessing, or MEG SLURM scripts exist.

**Fix:** Create `slurm/` directory; move and rename scripts; add evaluation and preprocessing jobs.

---

## WARNING (High Priority — Address Before Next Experiment Run)

### W1. SLURM Partition/Account Mismatch
**Agent:** slurm-reviewer
**File:** `CLIP-HBA/train_mem.slurm`

Uses `--partition=interactive_gpu` (wrong — for debugging), `--account=dsi_dgx_iacc`, `--qos=dgx_iacc`. CLAUDE.md specifies `batch_gpu` / `brain_ai` / `normal`. 30-hour job on interactive partition risks unexpected preemption.

**Fix:** Update to `--partition=batch_gpu --account=brain_ai --qos=normal` or document the cluster-specific overrides explicitly.

---

### W2. SLURM Trains Only Folds 2–3 (Not All 5)
**Agent:** slurm-reviewer
**File:** `CLIP-HBA/train_mem.slurm:68`

Comment on line 21 says "Trains all 5 LaMem folds sequentially" but loop is `for FOLD in 2 3`.

**Fix:** Change to `for FOLD in 1 2 3 4 5` or update the comment.

---

### W3. Hardcoded Absolute User Path in SLURM
**Agents:** slurm-reviewer, architecture-auditor
**File:** `CLIP-HBA/train_mem.slurm:40`

`cd /panfs/accrepfs.vampire/home/jenkm22/CLIP-HBA-Mem/CLIP-HBA` will fail for any other user or after a directory change.

**Fix:** Use `$(dirname "$(readlink -f "$0")")` or `--chdir` flag in `sbatch`.

---

### W4. /tmp Race Condition in SLURM Data Caching
**Agent:** slurm-reviewer

`LOCAL_DATA=/tmp/lamem_cache` is shared across concurrent jobs on the same node.

**Fix:** Use `$TMPDIR` (SLURM-allocated) or append job ID: `/tmp/lamem_cache_${SLURM_JOB_ID}`.

---

### W5. Image Resize Inconsistency
**Agent:** pipeline-integrity-checker
**File:** `CLIP-HBA/functions/train_mem_pipeline.py:43`

`MemDataset` uses `transforms.Resize((224, 224))` — distorts aspect ratio. `PerceptCLIPDataset` and standard CLIP use `Resize(224)` + `CenterCrop(224)`.

**Fix:** Update MemDataset to match standard CLIP preprocessing.

---

### W6. DoRA Bias Always Frozen Regardless of Parameters
**Agent:** ml-correctness-auditor
**File:** `CLIP-HBA/functions/train_behavior_things_pipeline.py:356–357`

`switch_dora_layers()` unconditionally sets `child.bias.requires_grad = False` regardless of `freeze_all` and `dora_state` parameters.

**Fix:** Gate bias freezing on the same condition as other DoRA parameters.

---

### W7. `sys.stdout` Manipulation Instead of `logging` Module
**Agents:** code-quality-inspector, architecture-auditor
**File:** `CLIP-HBA/functions/train_mem_pipeline.py:412–429`

Training redirects `sys.stdout` to a log file — fragile, breaks concurrent use, swallows exceptions. 29+ `print()` calls throughout; zero use of Python `logging`.

**Fix:** Replace with `logging.getLogger(__name__)` and file handlers; remove stdout redirection.

---

### W8. All File Paths Use `os.path.join` Instead of `pathlib.Path`
**Agents:** All agents

CLAUDE.md explicitly mandates `pathlib.Path`. ~100+ locations across all Python files use `os.path.join()` or string concatenation.

**Fix:** Systematic migration to `pathlib.Path`; add linting rule to catch regressions.

---

### W9. Destructive File Operations Without Backup
**Agent:** io-auditor
**Files:** `inference_meg_individual_pipeline.py:118`, `inference_behavior_pipeline.py:317`, `inference_meg_group_pipeline.py:374–376`

`shutil.rmtree(config['save_folder'])` with no backup or confirmation.

**Fix:** Rename old folder to `save_folder_backup_<timestamp>` or prompt before deletion.

---

### W10. Three Identical CLIP Variant Packages
**Agent:** architecture-auditor
**Dir:** `src/models/CLIPs/`

`clip/`, `clip_hba/`, `clip_hba_meg/` each contain byte-for-byte identical `simple_tokenizer.py` and nearly-identical `clip.py`.

**Fix:** Create `src/models/CLIPs/base/` with shared code; have variants import and extend.

---

### W11. History CSV Format Not Standardized Across Pipelines
**Agent:** experiment-tracking-reviewer

Memorability pipeline: `[epoch, train_loss, val_loss, spearman_rho, pred_std]`. Behavior pipeline: interleaved rows with `type: Train/Test`, different column names.

**Fix:** Define a `TrainingHistory` dataclass; all pipelines write the same schema.

---

### W12. Sweep CSV Write Has No File Lock
**Agent:** io-auditor
**File:** `CLIP-HBA/train_mem_sweep.py:169–172`

Appends to `sweep_results.csv` without `fcntl.flock` or equivalent. Concurrent sweep jobs could corrupt the file.

**Fix:** Add file locking around all CSV append operations.

---

### W13. Conda Environment Path Hardcoded in SLURM
**Agent:** slurm-reviewer
**File:** `CLIP-HBA/train_mem.slurm`

`source ~/envs/clip_hba/bin/activate` breaks for other users or relocated environments.

**Fix:** Use `conda activate clip_hba` after `conda init bash`, or pass the environment path as a parameterizable variable.

---

## SUGGESTION (Lower Priority — Improve Before Next Major Feature)

| ID | Issue | Agent | File/Location |
|----|-------|-------|---------------|
| S1 | Create `aggregate_cv_results.py` + `metrics.json` per run | experiment-tracking-reviewer | New file |
| S2 | Define normalization constants in a constants module | code-quality-inspector | New `utils/constants.py` |
| S3 | Decompose long functions (>50 lines) | code-quality-inspector | `train_mem_sweep.py`, `inference_meg_group_pipeline.py` |
| S4 | Add type hints to all public functions | code-quality-inspector, architecture-auditor | All pipeline files |
| S5 | Extract device setup to `utils/device.py::get_device()` | code-quality-inspector | 4+ files |
| S6 | Create `BaseImageDataset` for shared preprocessing | code-quality-inspector, architecture-auditor | 6 dataset variants |
| S7 | Move sweep configs (`BASE_CONFIG`, `SWEEP_GRID`) to YAML | code-quality-inspector | `configs/mem_sweep.yaml` |
| S8 | Rename `clip_lora_model` → `ClipLoRAModel` (PEP 8) | code-quality-inspector | `train_mem_pipeline.py:117` |
| S9 | Remove redundant `CUBLAS_WORKSPACE_CONFIG` assignment | ml-correctness-auditor | `train_behavior_things_pipeline.py:35` |
| S10 | Add runtime `assert emb.shape[-1] == 768` after encode | pipeline-integrity-checker | `train_mem_pipeline.py` |
| S11 | Use UUID suffix instead of timestamp in checkpoint names | io-auditor | `train_mem_pipeline.py:414` |
| S12 | Consolidate all outputs under `results/` hierarchy | experiment-tracking-reviewer | Directory structure |
| S13 | Log checkpoint path back into config JSON | experiment-tracking-reviewer | `train_mem_pipeline.py` |
| S14 | Move `Data/Encoder_Correspondence/*.py` to `analysis/` | architecture-auditor | `CLIP-HBA/Data/` |
| S15 | Increase SLURM `--mem` from 32G to 48–64G | slurm-reviewer | `train_mem.slurm` |
| S16 | Add `DoRALayer.__init__` explicit freeze of original layer | ml-correctness-auditor | `train_behavior_things_pipeline.py` |
| S17 | Explicitly set `nn.MSELoss(reduction='mean')` for clarity | ml-correctness-auditor | `train_mem.py:40` |
| S18 | Add SHA256 verification for large loaded checkpoints | io-auditor | `inference_mem.py` |
| S19 | Use HDF5 instead of `.npy` for large RDM arrays | io-auditor | `inference_meg_group_pipeline.py` |
| S20 | Add `--sweep_parent_id` flag for config lineage tracking | experiment-tracking-reviewer | `train_mem.py` |

---

## Summary

| Severity | Count |
|---|---|
| **CRITICAL** | **8** |
| **WARNING** | **13** |
| **SUGGESTION** | **20** |

---

## Recommended Immediate Action Order

```
1.  Fix checkpoint save format (C1)           — required for all future training
2.  Audit and fix normalization (C2)          — silently corrupts every model run
3.  Fix unseeded random_split (C5)            — breaks behavior training reproducibility
4.  Pin requirements / create env.yml (C6)   — needed before cluster deployment
5.  Fix SLURM fold loop 2–3 → 1–5 (W2)       — likely causing missing CV results
6.  Fix SLURM partition/account (W1)          — job may be preempted unexpectedly
7.  Fix SLURM hardcoded absolute path (W3)    — breaks on any other user/machine
8.  Create aggregate_cv_results.py (C4)       — needed to fill results table
9.  Extract shared utils / kill duplication (C3) — prerequisite for most refactoring
10. Add git hash + seed logging (C7)          — closes reproducibility gap
```

---

## ML Correctness Findings (PASSING)

The following critical constraints were verified as **correctly implemented**:

- **Frozen backbone:** `requires_grad=False` on all backbone params at `__init__`; `train()` override enforces `backbone.eval()`; forward pass uses `torch.no_grad()`
- **Optimizer targets only MLP head:** `mlp_parameters()` method used for optimizer construction
- **Validation hygiene:** `model.eval()` + `torch.no_grad()` consistently applied
- **LaMem splits predetermined:** Per-fold train/val/test CSVs loaded from disk, not regenerated
- **Metric calculations:** Spearman ρ computed on full prediction arrays; MSE aggregation correct
- **Loss function:** MSELoss with correct mean reduction and batch-weighted aggregation
- **DataLoader worker seeding:** `_seed_worker()` + seeded `Generator` passed to training DataLoader
- **No circular imports detected**
