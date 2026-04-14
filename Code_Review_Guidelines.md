# Code Review Checklist

An in-depth checklist for reviewing the CLIP-HBA-Mem repository. Adjust as needed for your specific review scope.

Sources: Google Engineering Practices, Microsoft Code Review Guidelines, NeurIPS Reproducibility Checklist, Papers With Code ML Completeness Checklist, PEP 8 / PyCQA, Wilson et al. "Best Practices for Scientific Computing", The Turing Way.

---

## 1. Documentation

### README Completeness
- Clearly states the purpose, goals, and scope of the project
- Provides a concise overview of project features or modules
- Includes information on how to set up, install, and run the project locally
- Describes any external services or APIs used (if applicable)
- Includes a "Results" section with expected outputs

### Supplementary Documentation
- Additional docs (e.g., `docs/` folder, gdrive models) are current and consistent
- Instructions for deployment or production setup, if relevant

### Usage Examples
- Clear instructions on how to run typical use cases or workflows
- Results tables reproducible from provided scripts
- Pre-trained models available with download instructions

---

## 2. Setup & Environment

### Environment Specification
- An `environment.yml` or `requirements.txt` for all dependencies
- Dependencies pinned to exact versions
- No unused packages in dependency files
- License compatibility verified across dependencies
- Dockerfile or Singularity file provided for full environment reproducibility

### Build & Run Instructions
- Build instructions or scripts (`setup.py`, `Makefile`, etc.) are easy to follow
- Steps to run and test the application locally are documented
- Makefile or equivalent exists for common operations (train, evaluate, test)

### Security & Dependencies
- No hardcoded secrets, API keys, or absolute user paths
- `torch.load` uses `weights_only=True` for untrusted data; no unsafe `pickle.load`
- Dependencies audited for known vulnerabilities
- `.gitignore` properly excludes sensitive files (`.env`, credentials, etc.)

---

## 3. Code Readability

### Coding Style & Consistency
- Consistent naming conventions (snake_case for functions/variables, PascalCase for classes)
- New code follows patterns already established in the repo
- No wildcard imports (`from x import *`); proper import ordering (stdlib / third-party / local)
- No circular imports
- No mutable default arguments (e.g., `def foo(x=[])`)

### Type Safety & Linting
- Type hints on all public function signatures (PEP 484/526)
- Automated linting configured (ruff, flake8, or pylint) and enforced in CI
- Pre-commit hooks configured for formatting and linting

### Comments & Docstrings
- Functions and classes have concise docstrings explaining purpose, parameters, and return values
- Comments explain "why," not "what"

### Modular & Reusable Code
- Code is organized into logical modules, classes, or files
- No duplicated logic that should be extracted into shared utilities
- Functions are appropriately sized (flag functions longer than ~50 lines)

### Logical Structure
- Well-structured folder hierarchy (e.g., `src/`, `tests/`, `utils/`)
- Proper `__init__.py` files where needed
- All file paths use `pathlib.Path`, not string concatenation

---

## 4. Key Parameters & Configuration

### Parameter Documentation
- Key parameters and configuration options are documented
- All hyperparameters specified, including defaults that were not changed
- Magic numbers replaced with named constants or config values

### Parameter Validation
- Code validates input parameters to avoid unexpected errors
- Invalid configurations produce clear error messages

### Parameter Visibility
- Critical parameters are easy to find and modify
- Config values not hardcoded inline (use config files or argparse)

### Reproducibility (NeurIPS Checklist, Papers With Code, Wilson et al.)

#### Random Seeds & Determinism
- All scripts set seeds for `torch`, `numpy`, `random`, and `torch.cuda`
- `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` set where needed
- Seeds documented and logged with every run
- Seeding happens before any data loading or model initialization

#### Compute Budget
- GPU type and count documented (e.g., 1x NVIDIA A6000)
- Approximate wall-clock training time per fold / full run reported
- Total GPU hours reported for full experiment (training + evaluation)
- Hardware requirements stated so others know what is needed to reproduce

#### Statistical Significance
- Results reported with variance across runs or folds (mean +/- std)
- Not single-run cherry picks; number of runs/folds clearly stated
- Comparison baselines evaluated under the same protocol and splits
- Negative results and failed configurations documented where relevant

#### Result Traceability
- Every result traceable to the exact code (git commit hash), data, and parameters that produced it
- Results regenerable from a single command using logged configs alone
- Config files or argparse defaults capture the full set of hyperparameters (including unchanged defaults)
- No manual steps required between "clone repo" and "reproduce result"

#### Dataset Versioning
- Dataset version or content hash tracked and logged per run
- Train/val/test split files saved, versioned, and committed (not regenerated randomly)
- Any preprocessing or filtering steps scripted and reproducible
- Raw data is immutable and never overwritten by pipeline steps

#### Environment Reproducibility
- Conda environment with a pinned `environment.yml` or `requirements.txt` capturing exact versions
- SLURM scripts activate the correct conda environment and load required modules

### Experiment Tracking
- Every run logs: config/hyperparams, train loss, val loss, metrics, epoch, and wall time
- Training logs include loss curves, learning rate, and gradient norms
- W&B (or equivalent) logs: config, git commit hash, random seeds, final metrics, and artifact paths
- Cross-validation results aggregated with mean +/- std
- Uses `logging` module, not `print()` for production code paths

---

## 5. Functionality

### ML Correctness
- Frozen backbone (`CLIP-HBA-Behavior`) has `requires_grad=False` enforced at model init
- Memorability head is the only component receiving gradient updates
- Loss function implementation correct (reduction, normalization)
- `model.eval()` and `torch.no_grad()` used during validation
- Metrics computed on the correct set with documented aggregation method (micro/macro)
- No `log(0)`, division by zero, or NaN/Inf propagation paths
- Gradient clipping configured if training is unstable

### Data Integrity
- Data loading is deterministic given a seed
- Augmentations applied only to training data
- Normalization statistics (mean/std) computed only from training data
- Image normalization matches CLIP's expected preprocessing exactly
- Train/val/test splits are predetermined, saved, and versioned
- No data leakage across splits; sample count verification (train + val + test = total)
- Raw data is immutable and never overwritten

### Checkpointing & Resumability
- Checkpoints save: epoch, model state dict, optimizer state, scheduler state, config, val metric, RNG states, and timestamp
- Training can resume from checkpoint after interruption (including SLURM preemption)
- Best model selection criteria documented and logged
- Model checkpoints linked to tracking runs

### Error Handling
- Code gracefully handles runtime exceptions
- Specific exceptions caught (no bare `except:` clauses)
- Errors logged or re-raised, never silently swallowed
- Failed/stopped runs leave enough state to diagnose what happened

### Performance & Resources
- Check for obvious bottlenecks or inefficiencies
- Context managers (`with` statements) used for files and connections
- No resource leaks (open files, unreleased GPU memory)
- Large tensors use appropriate formats (HDF5 or memmap over `.npy` when beneficial)

### Edge Cases
- Code handles unexpected input and invalid configurations properly
- Assertions and defensive checks at critical points

---

## PR & Review Process
- PRs are small and focused on a single concern
- PR description explains "why," not just "what"
- Breaking changes flagged explicitly
- Distinguish blocking vs. non-blocking feedback
- CI must pass before merge
