# CLIP-HBA-Mem — Project Context for Claude Code

## Project Overview
This repository contains **CLIP-HBA-Mem**, a memorability prediction model built on a
frozen **CLIP-HBA-Behavior** backbone. The project lives at the intersection of computer
vision, cognitive neuroscience, and machine learning, with the goal of predicting and
explaining image memorability using semantically grounded visual representations.

**Lab:** Tovar Brain Inspired AI Lab, Vanderbilt University   

---

## Repository Structure
# TODO: Paste your actual directory tree here. Example template below:
```
clip-hba-mem/
├── CLAUDE.md
├── README.md
├── requirements.txt / environment.yml       # TODO: confirm which
├── configs/                                  # hyperparameters, sweep configs
├── data/
│   ├── loaders/                              # dataset classes, augmentations
│   └── splits/                              # train/val/test split files
├── models/
│   ├── backbone/                            # CLIP-HBA-Behavior (FROZEN)
│   ├── head/                                # memorability prediction head
│   └── clip_hba_mem.py                      # full model assembly
├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── cross_validate.py
├── interpretability/
│   └── spose_pipeline.py                    # SPoSE dimension attribution
├── fmri/
│   ├── glm/                                 # GLM beta extraction (ds006883)
│   ├── rdm/                                 # RDM computation
│   └── noise_ceiling.py                     # noise ceiling via RDMs
├── slurm/                                   # all SLURM job scripts
├── notebooks/                               # exploration, figures
└── results/                                 # outputs, checkpoints, logs
```

---

## Critical Project Invariants
These are non-negotiable constraints that must hold at all times.
Any agent or review should flag violations as **CRITICAL**.

1. **Frozen backbone**: `CLIP-HBA-Behavior` must always have `requires_grad=False`
   on all parameters. No gradient should flow through it during training.
   The memorability head is the only component being optimized.

2. **Cross-validation correctness**: Splits must be predetermined and consistent
   across runs. No data leakage — memorability scores or image-level features
   from val/test sets must never influence training.

3. **Reproducibility**: All runs must set seeds for `torch`, `numpy`, and `random`.
   SLURM scripts must activate the correct conda environment before execution.

---

## Stack & Key Libraries
- **PyTorch** — model definition, training loop
- **CLIP** (ViT-L/14) — backbone base
- **NumPy / SciPy** — RSA, RDM computation, stats
- **scikit-learn** — cross-validation scaffolding, metrics
- **SLURM** — job scheduling on ACCRE

---

## Coding Conventions
# TODO: Add any project-specific conventions. Starters below:
- Type hints expected on all function signatures
- Docstrings required for all public functions and classes
- Config values must not be hardcoded inline — use config files or argparse
- Checkpoint saving must include: epoch, model state, optimizer state, config, val metric
- All file paths must be constructed via `pathlib.Path`, not string concatenation

---

## Named Sub-Agents

Use these agents by referencing their handle in your Claude Code prompt, e.g.:
`claude "@architecture-auditor review the full repo"`

---

### @architecture-auditor
**Scope:** Repository-wide structure and design  
**Task:**
- Map all modules, their responsibilities, and import dependencies
- Identify circular imports, dead code, or orphaned files
- Check that model components (backbone, head) are cleanly
  separated with no inappropriate cross-dependencies
- Verify that `configs/`, `data/`, `models/`, `training/`, and `results/` follow
  a logical and consistent structure
- Flag any logic that belongs in a utility module but is copy-pasted across files

**Output:** Findings report with severity: CRITICAL / WARNING / SUGGESTION

---

### @ml-correctness-auditor
**Scope:** Training logic, evaluation logic, and ML correctness  
**Task:**
- Verify the backbone (`CLIP-HBA-Behavior`) has `requires_grad=False` on all parameters
  and that this is enforced at model init, not just at the start of training
- Confirm the memorability head is the only component receiving gradient updates
- Check loss function implementation for correctness (reduction, normalization)
- Audit metric calculations (correlation, MSE, etc.) for off-by-one errors or
  incorrect aggregation across batches
- Check that `model.eval()` and `torch.no_grad()` are used correctly during validation
- Verify there is no label leakage between train/val/test in any preprocessing step
- Confirm cross-validation folds are constructed before any fitting/normalization

**Output:** Findings report with severity: CRITICAL / WARNING / SUGGESTION

---

### @pipeline-integrity-checker
**Scope:** End-to-end data flow  
**Task:**
- Trace data from raw image input through backbone, through head, to memorability prediction
- Verify tensor shapes and dtypes are consistent at every stage (log expected shapes)
- Check that image normalization (mean/std) matches CLIP's expected preprocessing exactly
- Flag any implicit shape assumptions (e.g., hardcoded batch sizes, unsqueeze hacks)

**Output:** Findings report with severity: CRITICAL / WARNING / SUGGESTION

---

### @reproducibility-agent
**Scope:** Reproducibility and environment hygiene  
**Task:**
- Check that all scripts set seeds for `torch`, `numpy`, `random`, and `torch.cuda`
  (including `torch.backends.cudnn.deterministic`)
- Verify `requirements.txt` or `environment.yml` exists and is up to date
- Check for hardcoded absolute paths that would break on another machine or cluster
- Confirm that SLURM scripts activate the correct conda environment and load modules
- Check that train/val/test split files are saved and versioned (not regenerated randomly)
- Verify that W&B (or equivalent) logs: config, git hash/commit, random seeds,
  final metrics, and artifact paths

**Output:** Findings report with severity: CRITICAL / WARNING / SUGGESTION

---

### @slurm-reviewer
**Scope:** All files in `slurm/`  
**Task:**
- Verify resource requests are appropriate: GPU count, CPU count, memory, time limit
  for the job type (training vs. evaluation vs. preprocessing)
- Confirm `--account=brain_ai`, `--partition=batch_gpu`, `--qos=normal` are set correctly
- Check that each job activates the conda environment before running Python
- Verify that checkpoint saving logic exists so jobs can resume after preemption
- Check that stdout/stderr are routed to named log files (not default slurm-%j.out only)
- Flag any jobs that request excessive resources relative to their task
- Check for missing `--mail-type` or notification flags if long-running jobs are involved

**Output:** Findings report with severity: CRITICAL / WARNING / SUGGESTION

---

### @io-and-storage-auditor
**Scope:** File I/O, checkpointing, logging  
**Task:**
- Check that all file paths use `pathlib.Path` and are configurable (not hardcoded)
- Verify checkpoints save: epoch, model state dict, optimizer state, config dict,
  val metric, and timestamp
- Confirm that results, logs, and checkpoints are organized in a consistent,
  human-navigable directory structure under `results/`
- Check for potential race conditions in multi-job SLURM scenarios
  (e.g., multiple jobs writing to the same results file)
- Flag any large tensors or embeddings being saved as `.npy` when HDF5 or memmap
  would be more appropriate
- Verify W&B artifact logging (or equivalent) is capturing model checkpoints

**Output:** Findings report with severity: CRITICAL / WARNING / SUGGESTION

---

### @code-quality-inspector
**Scope:** All Python files  
**Task:**
- Flag functions longer than ~50 lines that should be decomposed
- Identify duplicated logic across files that should be extracted into shared utilities
- Check for missing type hints on public functions and classes
- Identify magic numbers that should be named constants or config values
- Check for inconsistent naming conventions (snake_case, variable names, etc.)
- Flag bare `except` clauses or overly broad exception handling
- Identify any `print()` debugging left in production code paths
  (should be `logging` module)

**Output:** Findings report with severity: CRITICAL / WARNING / SUGGESTION

---

### @experiment-tracking-reviewer
**Scope:** Logging, W&B integration, results organization  
**Task:**
- Verify that every training run logs: config/hyperparams, train loss, val loss,
  val memorability correlation, epoch, and wall time
- Confirm that model checkpoints are linked to W&B runs (or equivalent tracking)
- Check that cross-validation results are aggregated and logged with mean ± std
- Flag any experiment whose results cannot be reconstructed from logged artifacts alone

**Output:** Findings report with severity: CRITICAL / WARNING / SUGGESTION

---

## Full Audit Command
To run a comprehensive sweep across all agents simultaneously:

```bash
claude "Run all named agents in parallel: @architecture-auditor, @ml-correctness-auditor,
@pipeline-integrity-checker, @reproducibility-agent, @slurm-reviewer,
@io-and-storage-auditor, @code-quality-inspector, @experiment-tracking-reviewer.
Each agent should produce a findings report with severity levels CRITICAL, WARNING,
and SUGGESTION. After all agents complete, synthesize results into a single
prioritized action list, grouped by severity."
```

---

## Notes for Collaborators
- The Bainbridge collaboration centers on memorability scoring methodology —
  ensure any ResMem comparison uses the same image preprocessing and scoring protocol
- SPoSE dimension indices must be documented and pinned to a specific embedding version
- ds006883 (O'Doherty et al.) GLM parameters should match published preprocessing pipelines