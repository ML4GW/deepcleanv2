# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepClean v2 is a gravitational wave noise subtraction pipeline for LIGO detectors. It uses convolutional autoencoders to subtract noise couplings (60Hz, 120Hz, etc.) from strain data, orchestrated by Law/Luigi and deployed via Apptainer containers on HPC clusters.

## Commands

### Setup

```bash
git submodule update --init --recursive
poetry install
```

### Linting / Formatting

```bash
pre-commit run --all-files   # isort + black + flake8
black .                      # format (line length 79)
isort .                      # sort imports
flake8 .                     # lint
```

Config: line length 79 (`.flake8` and `pyproject.toml`), isort `known_first_party = ["deepclean", "train", "utils", "infer", "export"]`.

### Running the Pipeline

Each stage runs via Law/Luigi and executes inside an Apptainer container:

```bash
source export_vars.sh   # set DEEPCLEAN_CONTAINER_ROOT, DATA_DIR, RESULTS_DIR, IFO, PROBLEM

# Build containers (per project)
cd projects/train && apptainer build $DEEPCLEAN_CONTAINER_ROOT/train.sif apptainer.def

# Fetch data
LAW_CONFIG_FILE=luigi.cfg poetry run law run deepclean.tasks.Fetch --image data.sif ...

# Train
LAW_CONFIG_FILE=luigi.cfg poetry run law run deepclean.tasks.Train --image train.sif --gpus 0 ...

# Infer/Clean
LAW_CONFIG_FILE=luigi.cfg poetry run law run deepclean.tasks.Clean --image clean.sif --gpus 0 ...
```

Dev mode (maps local code into container without rebuilding):

```bash
poetry run law run deepclean.tasks.Train --dev ...
```

### Tests

Tests live in `microservice/` (the legacy submodule):

```bash
cd microservice
pytest libs/architectures/tests
pytest libs/trainer/tests
pytest libs/infer/tests
```

## Architecture

### Task / Container Layering

The pipeline has three execution stages, each with its own Poetry project and Apptainer container under `projects/`:

| Stage | Project dir | Law task | Container |
|---|---|---|---|
| Data fetching | `projects/data/` | `deepclean.tasks.Fetch` | `data.sif` |
| Training | `projects/train/` | `deepclean.tasks.Train` | `train.sif` |
| Inference | `projects/infer/` or `projects/clean/` | `deepclean.tasks.Clean` | `clean.sif` |

`deepclean/tasks/` wires these Law tasks together and handles HTCondor submission. `deepclean/base.py::DeepCleanTask` is the base class that mounts GPU (`--nv`), binds cluster filesystems (`/cvmfs`, `/gpfs`, `/ceph`, etc.), and forwards credentials.

### Couplings System

`deepclean/couplings/` defines which noise frequencies to subtract and which witness channels to use. Each file defines a `SubtractionProblem` subclass (e.g., `Sub120Hz`) that specifies a frequency band and a list of witness channel names. The `luigi.cfg` `[deepclean]` section selects the active `ifo` and `problem` at runtime.

### Training Internals (`projects/train/train/`)

- **`cli.py`** — `LightningCLI` entry point; links `num_witnesses`, `sample_rate`, and frequency band args across components
- **`data.py`** — `DeepCleanDataset` (Lightning DataModule): reads HDF5, applies bandpass filter, batches with stride
- **`model.py`** — `DeepClean` Lightning Module wrapping architecture + loss
- **`architectures/autoencoder.py`** — Conv1d encoder (stride-2 downsampling) / decoder (upsampling) with BatchNorm + Tanh
- **`metrics.py`** — `PsdRatio` / `OnlinePsdRatio`: loss computed as power spectral density ratio in the target frequency band

Training config is `projects/train/config.yaml` (YAML) and can be overridden via CLI.

### Data Format

- LDG → GWF (raw LIGO frame files)
- Fetched → HDF5 (ML-ready)
- Cleaned output → GWF frames (1 second each)

### Key Configuration

- `luigi.cfg` — selects IFO, problem, container root, strain channel, scheduler settings
- `law.cfg` — Singularity sandbox settings
- `export_vars.sh` — template for environment variables (must be sourced before running pipeline)
- `projects/train/config.yaml` — model hyperparameters, optimizer, trainer settings

### Toolbox Submodules

`toolbox/` contains git submodules used as library dependencies:
- `ml4gw/` — shared ML utilities (autoencoders, transforms, dataloading)
- `hermes/` — serialization/serving
- `mldatafind/` — data discovery

`microservice/` is a legacy submodule (IaaS implementation, being phased out).
