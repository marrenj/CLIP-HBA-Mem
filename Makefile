.PHONY: install train-mem train-behavior test lint format

# Install the package in editable mode plus all runtime dependencies.
install:
	pip install -e .
	pip install -r requirements.txt

# Run memorability training (configure via environment variables; see train_mem.py).
train-mem:
	cd CLIP-HBA && python train_mem.py

# Run behavioral fine-tuning (Things dataset).
train-behavior:
	cd CLIP-HBA && python train_behavior.py

# Run the full test suite (CPU-only; no data files required).
test:
	pytest tests/ -v --tb=short

# Check code style — exits non-zero on violations (used in CI).
lint:
	ruff check CLIP-HBA/ src/
	ruff format --check CLIP-HBA/ src/

# Auto-fix style issues in-place.
format:
	ruff format CLIP-HBA/ src/
	ruff check --fix CLIP-HBA/ src/
