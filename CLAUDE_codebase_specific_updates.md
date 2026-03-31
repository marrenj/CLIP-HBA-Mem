# CLAUDE.md — Codebase-Specific Updates
# ─────────────────────────────────────────────────────────────────────────────
# This file contains CORRECTIONS and REPLACEMENTS to the generic MEGMem additions
# document, derived from reading the actual source code.
#
# Priority: If anything here conflicts with the generic additions file,
# THIS FILE WINS. It is grounded in the actual code.
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CORRECTIONS TO PROJECT INVARIANTS
# Replace the generic MEGMem invariants with these code-verified versions.
# ══════════════════════════════════════════════════════════════════════════════

## CORRECT SLURM Configuration (replaces generic version)
# The SLURM scripts in the repo use:
#   --account=dsi_dgx_iacc      (NOT brain_ai)
#   --partition=interactive_gpu  (NOT batch_gpu)
#   --qos=dgx_iacc               (NOT normal)
# Update the CLAUDE.md SLURM invariants and @slurm-reviewer accordingly.

## Correct DoRA Application Order (CRITICAL — must match checkpoint)
# Both the reference script (inference_meg_group_pipeline.py) and the
# extraction script (extract_embeddings_meg.py) apply DoRA in this exact order:
#
#   Step 1 — Text transformer FIRST:
#     apply_dora_to_ViT(model, n_vision_layers=0,  n_transformer_layers=1, r=32, dora_dropout=0.1)
#
#   Step 2 — Vision encoder SECOND:
#     apply_dora_to_ViT(model, n_vision_layers=24, n_transformer_layers=0, r=32, dora_dropout=0.1)
#
# Reversing this order changes which layers get which DoRA parameters and will
# silently corrupt the checkpoint load. Flag any deviation as CRITICAL.

## Correct Checkpoint Loading Pattern (from extract_embeddings_meg.py)
#   state_dict = torch.load(meg_checkpoint, map_location='cpu')
#   state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
#   model.load_state_dict(state_dict, strict=True)
#
# The REFERENCE SCRIPT (inference_meg_group_pipeline.py) uses:
#   model_state_dict = torch.load(config['model_path'])    # no map_location!
#   adjusted_state_dict = {key.replace("module.", ""): ...}
#   model.load_state_dict(adjusted_state_dict)             # strict defaults to True
#
# The extraction script adds map_location='cpu' which is a pragmatic improvement
# (avoids GPU OOM when loading on CPU-only nodes) and does NOT affect state dict
# content. The key difference to watch: strict=True in both. Any code that uses
# strict=False for the MEG backbone is a CRITICAL violation.

## Correct temporal parameters
# The MEG model covers: ms_start=-100, ms_end=1300, ms_step=5 → 281 timepoints
# Extraction uses: train_window_size=0 (exact per-timepoint parameters, no averaging)
# The reference inference script uses: train_window_size=15
# This discrepancy is INTENTIONAL: extraction needs exact per-timepoint weights,
# inference uses a smoothed window. The @meg-checkpoint-fidelity-auditor should
# document this distinction, not flag it as a bug.

## Correct MLP head for MEGMem
# MEGMem uses MLPOnlyHead (from functions/train_mem_pipeline.py) with:
#   input_dim=66  (CLIP-HBA-MEG embedding dimensionality — NOT 768)
#   hidden_dims defaults: (32, 16) in train_mem_meg.py, [32, 16] in inference_mem_meg.py
#   No output activation (unbounded regression — correct for memorability)
#   mlp_parameters() returns ALL parameters (MLPOnlyHead has no frozen backbone)
#
# The optimizer is built using mlp_parameters(), which for MLPOnlyHead returns
# list(self.mlp_head.parameters()) — i.e., all params. This is correct since
# there is no backbone in this model class.

## Image normalization (both pipelines — verified consistent)
# Both MemDataset and the MEG ImageDataset use:
#   mean=[0.52997664, 0.48070561, 0.41943838]
#   std=[0.27608301, 0.26593025, 0.28238822]
# This is the CLIP-HBA normalization, distinct from standard CLIP normalization.
# PerceptCLIPDataset uses standard CLIP normalization — do NOT mix these.


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — KNOWN BUG: classnames66 extraction in inference_meg_group_pipeline.py
# ══════════════════════════════════════════════════════════════════════════════

## BUG — classnames extraction discrepancy (CRITICAL)
#
# In inference_meg_group_pipeline.py (the canonical reference):
#   classnames = [x[0] for x in classnames66]   ← extracts FIRST CHARACTER of each string
#
# In extract_embeddings_meg.py (the extraction script):
#   classnames = classnames66                    ← passes full strings directly
#
# Because classnames66 in spose_dimensions.py is a plain list of strings
# (e.g. ['metallic; artificial', 'food-related', ...]), doing [x[0] for x in classnames66]
# produces ['m', 'f', 'a', ...] — the first CHARACTER of each label, not the label itself.
#
# This means:
#   - The reference inference script tokenizes single-character class prompts
#   - The training pipeline and extraction script tokenize the full semantic labels
#
# The model checkpoint was trained with full labels (train_meg_things_pipeline.py uses
# classnames = classnames66 directly). The reference script's [x[0] for x in classnames66]
# appears to be a leftover from when classnames66 was a list of tuples, not strings.
#
# IMPACT: Running run_meg_group_inference() with the current spose_dimensions.py
# produces different text embeddings than what the model was trained with.
# This is a silent correctness bug.
#
# CORRECT USAGE: classnames = classnames66   (pass list of strings directly)
# OR:            classnames = [x for x in classnames66]  (identity, but explicit)
#
# The @meg-checkpoint-fidelity-auditor MUST check which classnames pattern your
# MEGMem code uses. Use classnames66 directly, matching the training pipeline.


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ACTUAL REPOSITORY STRUCTURE (replaces the TODO tree)
# ══════════════════════════════════════════════════════════════════════════════

```
CLIP-HBA-Mem/
├── CLAUDE.md
├── CLIP-HBA/
│   ├── functions/
│   │   ├── inference_behavior_pipeline.py   # CLIP-HBA-Behavior inference
│   │   ├── inference_meg_group_pipeline.py  # ← CANONICAL MEG reference (read-only)
│   │   ├── inference_meg_individual_pipeline.py
│   │   ├── spose_dimensions.py              # classnames66 — list of 66 strings
│   │   ├── train_behavior_things_pipeline.py # seed_everything, CLIPHBA (behavior)
│   │   ├── train_meg_individual_pipeline.py
│   │   ├── train_meg_things_pipeline.py     # MEG group training
│   │   └── train_mem_pipeline.py            # CLIPHBAMem, CLIPFrozenMLP, MLPOnlyHead,
│   │                                        # EmbeddingDataset, MemDataset, run_mem_training
│   ├── extract_embeddings.py                # Pre-extract 768-dim CLIP-HBA embeddings
│   ├── extract_embeddings.slurm
│   ├── extract_embeddings.sh
│   ├── extract_embeddings_meg.py            # Pre-extract 66-dim MEG embeddings (281 timepoints)
│   ├── fix_pipeline.py                      # One-time patch script (already applied)
│   ├── inference_behavior.py
│   ├── inference_meg_group.py               # Calls run_meg_group_inference
│   ├── inference_meg_individual.py
│   ├── inference_mem.py                     # CLIP-HBA-Mem / CLIPFrozenMLP inference
│   ├── inference_mem_meg.py                 # MEGMem memorability inference (per-timepoint)
│   ├── train_behavior.py
│   ├── train_clip_mlp.py                    # CLIPFrozenMLP training
│   ├── train_clip_mlp.slurm
│   ├── train_clip_mlp_sweep.py              # Grid search for CLIPFrozenMLP
│   ├── train_meg_group.py
│   ├── train_meg_individual.py
│   ├── train_mem.py                         # CLIPHBAMem training
│   ├── train_mem.slurm
│   ├── train_mem_hyperparam_search.py       # Bayesian (Optuna TPE) sweep for CLIPHBAMem
│   ├── train_mem_meg.py                     # MEGMem MLP head training (one timepoint)
│   └── train_mem_meg_hyperparam_search.py   # Bayesian sweep for MEGMem MLP heads
├── src/
│   └── models/CLIPs/
│       ├── clip/           # Base CLIP (behavior)
│       ├── clip_hba/       # CLIP-HBA-Behavior (behavior fine-tuning)
│       └── clip_hba_meg/   # CLIP-HBA-MEG (MEG-tuned backbone) ← different model.py
│           ├── clip.py
│           ├── model.py    # ModifiedCLIP / ModifiedResNet with temporal weighting matrix
│           └── simple_tokenizer.py
└── Data/
    ├── lamem/
    │   ├── lamem_train_{1..5}.csv / lamem_val_{1..5}.csv / lamem_test_{1..5}.csv
    │   └── embeddings/      # Pre-extracted 768-dim .pt files (from extract_embeddings.py)
    │       ├── clip_hba_mem_fold{1..5}_{train,val,test}.pt
    │       └── clip_frozen_mlp_fold{1..5}_{train,val,test}.pt
    ├── combined_lamem_memcat/
    │   ├── lamem_memcat_{train,val,test}_split_{01..10}.csv
    │   ├── embeddings/      # Pre-extracted 768-dim .pt files
    │   └── meg_embeddings/  # Pre-extracted 66-dim MEG .pt files (from extract_embeddings_meg.py)
    │       └── clip_hba_meg_mem_tp{-100..1300}_fold{1..10}_{train,val,test}.pt
    ├── lamem/images/
    ├── memcat/images/
    └── memcat/memcat_image_data.csv
```


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — UPDATED NAMED SUB-AGENTS
# These are grounded in the actual codebase. Replace generic versions.
# ══════════════════════════════════════════════════════════════════════════════

---

### @meg-checkpoint-fidelity-auditor
**Scope:** `extract_embeddings_meg.py`, `inference_mem_meg.py`, `inference_meg_group.py`,
           and `functions/inference_meg_group_pipeline.py`
**Priority:** HIGHEST — run before all other MEG agents
**Task:**

Open `functions/inference_meg_group_pipeline.py` as the read-only canonical reference.
Compare your MEGMem code against it on every dimension below. Produce a side-by-side
diff table for each check. Flag deviations as CRITICAL.

**DoRA application order (CRITICAL):**
- Verify text transformer DoRA applied FIRST:
  `apply_dora_to_ViT(model, n_vision_layers=0, n_transformer_layers=1, r=32, dora_dropout=0.1)`
- Verify vision encoder DoRA applied SECOND:
  `apply_dora_to_ViT(model, n_vision_layers=24, n_transformer_layers=0, r=32, dora_dropout=0.1)`
- Flag if order is reversed — this silently corrupts the checkpoint load.

**Checkpoint loading (CRITICAL):**
- Verify `torch.load(checkpoint, map_location='cpu')` is used (safe for all hardware).
- Verify `{k.replace('module.', ''): v for k, v in state_dict.items()}` for key stripping.
- Verify `model.load_state_dict(state_dict, strict=True)` — strict must be True.
- Flag `strict=False` as CRITICAL.

**classnames66 usage (CRITICAL — known bug in reference):**
- The reference script incorrectly uses `classnames = [x[0] for x in classnames66]`
  which extracts first characters ('m', 'f', ...) from string labels.
- The training pipeline uses `classnames = classnames66` (full strings).
- Verify that your MEGMem code uses `classnames = classnames66` (full strings),
  matching the training pipeline — NOT the reference script's broken pattern.
- If the reference's broken pattern is present anywhere in your MEGMem code, flag CRITICAL.

**CLIPHBA instantiation parameters:**
- Verify: `backbone_name='ViT-L/14'`, `pos_embedding=True`, `ms_start=-100`,
  `ms_step=5`, `ms_end=1300`
- Verify `train_start=EXTRACT_START`, `train_step=extract_step`, `train_end=EXTRACT_END`
- Verify `train_window_size=0` during extraction (exact per-timepoint params, no averaging).
  The reference script uses `train_window_size=15` for inference — this discrepancy is
  INTENTIONAL. Document it clearly rather than flagging it.

**Forward pass:**
- Verify `model.eval()` called before inference.
- Verify `torch.no_grad()` wraps the forward pass.
- Verify the forward call is `pred_emb_3d, _, _ = model(images)` — the MEG model
  returns a 3-tuple `(pred_emb_3d, pred_rdm_3d, pred_feature_3d)`.
- Verify `pred_emb_3d.float().cpu()` is called for dtype safety.
- Verify shape indexing: `pred_emb_3d[t_idx]` gives `[batch, 66]` for timepoint t_idx.

**Output:** Side-by-side diff table + CRITICAL/WARNING/SUGGESTION findings

---

### @meg-mlp-head-auditor
**Scope:** `functions/train_mem_pipeline.py` (MLPOnlyHead), `train_mem_meg.py`,
           `inference_mem_meg.py`, `train_mem_meg_hyperparam_search.py`
**Task:**

**Architecture verification:**
- Verify `MLPOnlyHead` is used (not `CLIPHBAMem` or `CLIPFrozenMLP`).
- Verify `input_dim=66` — MEG embeddings are 66-dim, not 768-dim.
  Hardcoded 768 anywhere in the MEG training path is CRITICAL.
- Document the actual hidden_dims in use (defaults: `(32, 16)` in training,
  `[32, 16]` in inference — verify these match).
- Verify no output activation — `mlp_head` final layer is `nn.Linear(in_dim, 1)` with
  no sigmoid/tanh. Output activation would silently clip memorability predictions.
- Verify dropout toggles correctly via `model.train()` / `model.eval()`.

**Optimizer and gradient flow:**
- Verify `mlp_parameters()` is called to build the optimizer.
- For `MLPOnlyHead`, `mlp_parameters()` returns `list(self.mlp_head.parameters())` —
  all parameters, which is correct since MLPOnlyHead has no backbone.
- Verify no backbone parameters are inadvertently added to the optimizer.

**Embedding dataset:**
- Verify `EmbeddingDataset` is used, loading `.pt` files with key `'embeddings'` [N, 66].
- Verify the model_type string encodes the timepoint: `clip_hba_meg_mem_tp{timepoint_ms}`
  so that `_run_mem_training_impl` resolves the correct embedding file.
- Verify `input_dim=66` is passed in config — if missing, defaults to 768 (CRITICAL).

**Checkpoint saving/loading:**
- Verify checkpoint filename includes timepoint and timestamp for traceability.
- In `inference_mem_meg.py`, verify `_find_checkpoint` locates files matching
  `tp{timepoint_ms}_fold{fold}_*.pth` and returns the most recently modified.
- Verify `strict=True` when loading the MLP head checkpoint.

**Output:** Architecture documentation + CRITICAL/WARNING/SUGGESTION findings

---

### @meg-preprocessing-auditor
**Scope:** `functions/train_mem_pipeline.py` (MemDataset), `extract_embeddings_meg.py`,
           `functions/inference_meg_group_pipeline.py` (ImageDataset)
**Task:**

**Image preprocessing — verify consistency across all pipelines:**
- Both `MemDataset` (used in extraction) and `ImageDataset` (used in reference inference)
  apply: `Resize((224, 224))`, `ToTensor()`,
  `Normalize(mean=[0.52997664, 0.48070561, 0.41943838], std=[0.27608301, 0.26593025, 0.28238822])`
- Flag any code path that uses standard CLIP normalization
  `(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])`
  for MEG memorability images. `PerceptCLIPDataset` uses the standard CLIP normalization
  and must NEVER be used with the MEG backbone.
- Verify `Resize((224, 224))` not `Resize(224)` + `CenterCrop` — MemDataset uses the
  former; PerceptCLIPDataset uses the latter. These produce different crops for
  non-square images.

**Temporal extraction parameters:**
- Verify `MEG_MS_START=-100`, `MEG_MS_END=1300`, `MEG_MS_STEP=5` in `extract_embeddings_meg.py`.
- Verify `TRAIN_WINDOW_SIZE=0` is used during extraction (exact per-timepoint parameters).
- Verify extracted timepoints span -100 to 1300 ms at 5 ms resolution → 281 timepoints.
- Flag any extraction that uses a non-zero `train_window_size` — this would average
  neighbouring timepoints' learned parameters and would not produce exact per-timepoint features.

**Output:** Preprocessing chain documentation + CRITICAL/WARNING/SUGGESTION findings

---

### @meg-data-integrity-auditor
**Scope:** `train_mem_meg.py`, `train_mem_meg_hyperparam_search.py`,
           `functions/train_mem_pipeline.py` (EmbeddingDataset, run_mem_training)
**Task:**

**Embedding file resolution:**
- The model_type convention `clip_hba_meg_mem_tp{timepoint_ms}` must match the filenames
  produced by `extract_embeddings_meg.py`: `clip_hba_meg_mem_tp{tp}_fold{fold}_{split}.pt`.
- Verify `_run_mem_training_impl` correctly constructs the path:
  `emb_dir / f'{model_type}_fold{fold}_{split}.pt'`
- Verify that if `timepoint_ms=0`, the file is `clip_hba_meg_mem_tp0_fold1_train.pt` —
  NOT `clip_hba_meg_mem_tp+0_...` or any other format.

**Cross-validation setup:**
- MEGMem uses image-level cross-validation on LaMem/MemCat image memorability scores.
  This is different from the MEG backbone training (which uses RDMs and subject-level data).
  There is no subject-level CV required for MEGMem memorability — verify this is understood.
- Verify folds 1–5 for lamem, 1–10 for combined_lamem_memcat.
- Verify the CSV paths match: `lamem_memcat_{train,val,test}_split_{fold:02d}.csv`
  (zero-padded fold index for combined dataset).

**Sweep configuration:**
- In `train_mem_meg_hyperparam_search.py`, verify `train_fraction=0.5` is used during
  sweep trials (intentional — speeds up sweep) and that final training uses `train_fraction=1.0`.
- Verify `random_seed=42` for sweep trials and `random_seed=1` for final training runs —
  these must be documented so results are reproducible.
- Verify `save_checkpoint=False` during sweep to avoid filling disk with trial checkpoints.

**SLURM account/partition (from actual scripts):**
- Correct values (from train_mem.slurm, train_clip_mlp.slurm, extract_embeddings.slurm):
  `--account=dsi_dgx_iacc`, `--partition=interactive_gpu`, `--qos=dgx_iacc`
- Flag any script using `--account=brain_ai` or `--partition=batch_gpu` — these were
  incorrect in the CLAUDE.md template and must be updated.

**Output:** CRITICAL/WARNING/SUGGESTION findings

---

### @meg-vs-clip-alignment-auditor
**Scope:** Cross-modal consistency between MEGMem (66-dim, timepoint-resolved) and
           CLIP-HBA-Mem (768-dim, static)
**Task:**

**Ground-truth memorability label consistency:**
- Both MEGMem and CLIP-HBA-Mem train on `score` column from the same LaMem/MemCat CSVs.
- Verify that `EmbeddingDataset` in both pipelines loads scores from the same source
  `.pt` files — the scores are embedded at extraction time and must not drift.
- Verify that scores in MEG `.pt` files (`clip_hba_meg_mem_tp{tp}_fold{fold}_{split}.pt`)
  match scores in behavior `.pt` files (`clip_hba_mem_fold{fold}_{split}.pt`) for the
  same image at the same fold. Any mismatch means one was extracted from a different CSV.

**Metric consistency:**
- Both pipelines use Spearman ρ (`spearmanr`) via `evaluate_mem_model`.
- MEGMem produces 281 Spearman ρ values (one per timepoint) and per-fold values.
- Verify that summary CSVs aggregate fold results using mean ± std, not just mean.
- Verify that when comparing MEGMem vs. CLIP-HBA-Mem performance, both use the
  same test fold, not different folds or the full dataset.

**Temporal profile sanity:**
- The MEGMem Spearman ρ curve over time should be near-zero at -100 ms (pre-stimulus),
  rise through 100–500 ms, and have a characteristic peak shape consistent with
  cognitive neuroscience literature on memorability encoding timing.
- Flag a flat or monotonically rising curve as suspicious — may indicate data leakage
  or incorrect timepoint indexing.

**Output:** Alignment report + CRITICAL/WARNING/SUGGESTION findings

---

### @slurm-reviewer (UPDATED — corrects template errors)
**Scope:** All `.slurm` and `.sh` files
**Task:**

**Correct cluster configuration (verified from actual scripts):**
- Correct: `--account=dsi_dgx_iacc`, `--partition=interactive_gpu`, `--qos=dgx_iacc`
- Correct GPU type: `nvidia_a100-sxm4-40gb` (from actual scripts)
- Environment: `module load python/3.11.5` + `source ~/envs/clip_hba/bin/activate`
- Working directory: `/panfs/accrepfs.vampire/home/jenkm22/CLIP-HBA-Mem/CLIP-HBA`

**Image caching pattern (used in existing scripts):**
- Images are rsynced to `/tmp/lamem_cache` and `/tmp/memcat_cache` once per job.
- Verify new SLURM scripts follow this pattern to avoid repeated NFS reads.
- Verify cached copies are checked for existence before re-syncing (`if [ ! -d ... ]`).

**MEG-specific SLURM considerations:**
- `train_mem_meg.py` trains one timepoint at a time — SLURM array jobs are the
  correct parallelism strategy (one job per timepoint, parameterized by `TIMEPOINT_MS`).
- `extract_embeddings_meg.py` processes all 281 timepoints in a single job (memory-efficient
  single-pass approach). Verify time limit is sufficient (281 timepoints × N folds).
- The sweep (`train_mem_meg_hyperparam_search.py`) can run one timepoint per job.
  Verify `CUDA_DEVICE` environment variable is set correctly per job.

**Output:** CRITICAL/WARNING/SUGGESTION findings

---

### @code-quality-inspector (MEG-specific additions)
**Scope:** All MEG-related Python files
**Additional MEG-specific checks:**

- `fix_pipeline.py` is a one-time patch script that modifies `functions/train_mem_pipeline.py`.
  Verify whether it has already been applied — if the target string is not present in
  `train_mem_pipeline.py`, the patch has already been applied. The file should not be
  re-run. Flag it as a maintenance hazard if it is still in the repo without a clear
  "already applied" marker or if it is included in any automated pipeline.

- In `extract_embeddings_meg.py`, the variable `tp_desc` is defined in the
  `combined_lamem_memcat` branch but is never used in the `lamem` branch, and its
  f-string construction happens outside the if/else block on a variable that may be
  undefined. Verify this does not cause a `NameError` when `training_data='lamem'`.

- Verify that `n_tps` is defined before the print statement that uses it — in the
  current code, `n_tps` is only defined in the `combined_lamem_memcat` branch but
  used in a print statement that runs unconditionally. This is a latent `NameError`
  for `training_data='lamem'`.

- Flag the `train_window_size=15` comment in the reference script's `__main__` block
  — the comment says "Do not change train_window_size" but this is for inference only.
  The extraction script intentionally uses `train_window_size=0`. Ensure this
  distinction is documented in the code, not just in CLAUDE.md.

**Output:** CRITICAL/WARNING/SUGGESTION findings


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — UPDATED FULL AUDIT COMMAND
# ══════════════════════════════════════════════════════════════════════════════

## Full Audit Command (grounded in actual codebase)

```bash
# Step 1: MEG checkpoint fidelity — prerequisite for all other MEG agents
# Focus it on the classnames bug and DoRA order specifically.
claude "@meg-checkpoint-fidelity-auditor:
  1. Compare classnames66 usage in extract_embeddings_meg.py vs inference_meg_group_pipeline.py
  2. Verify DoRA application order (text first, vision second) in all MEG code paths
  3. Verify checkpoint loading: map_location='cpu', module. stripping, strict=True
  4. Compare forward pass: model returns 3-tuple, only pred_emb_3d[t_idx] is used for MEGMem
  Produce side-by-side diff table. Flag the classnames issue as CRITICAL."

# Step 2: All remaining agents in parallel
claude "Run the following agents in parallel and synthesize findings:
  @architecture-auditor,
  @ml-correctness-auditor,
  @pipeline-integrity-checker,
  @reproducibility-agent,
  @slurm-reviewer (note: correct account=dsi_dgx_iacc, partition=interactive_gpu),
  @io-and-storage-auditor,
  @code-quality-inspector (check fix_pipeline.py patch status; check n_tps NameError in extract_embeddings_meg.py),
  @experiment-tracking-reviewer,
  @meg-mlp-head-auditor (verify input_dim=66 everywhere, no output activation),
  @meg-preprocessing-auditor (verify normalization consistency; no PerceptCLIPDataset in MEG path),
  @meg-data-integrity-auditor (verify model_type string convention, fold CSV naming),
  @meg-vs-clip-alignment-auditor.

After all complete, synthesize into one prioritized action list. CRITICAL issues first.
Elevate to top of CRITICAL list: (1) classnames66 bug, (2) n_tps NameError,
(3) any use of input_dim=768 in MEG training path, (4) any strict=False checkpoint loads."
```

## Quick MEGMem-only audit (for active development sessions)
```bash
claude "Run @meg-checkpoint-fidelity-auditor and @meg-mlp-head-auditor in parallel.
Focus on:
  - classnames66 usage (full strings vs first characters)
  - input_dim=66 in all MEGMem training and inference configs
  - DoRA application order (text before vision)
  - strict=True on all MEG checkpoint loads
Report only CRITICAL findings."
```
