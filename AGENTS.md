# Agent instructions

## Project Overview

`radar-hail-probability` (hailathon) is a WIP modernization of a legacy weather radar data processing pipeline. The goal is to rewrite the production system in Python. The original pipeline generates hail probability fields — **POH** (Probability Of Hail) and **LHI** (Large Hail Index) — from IRIS radar composites and NWP data.

**Core values**: Modern tools and standards, code readability and maintainability, reuse over reimplementation. Where applicable, leverage `scipy`, `xarray`, `wradlib`, and other established libraries for mathematical and radar-specific operations instead of reimplementing algorithms. **Do it the pythonic way.** This is not strictly a rewrite project, but rather a replacement with comparable functionality and results.

## Legacy architecture

The legacy implementation (in `legacy/`) consists of:
- **C programs**: Core processing logic (POH extraction, LHI/RAE mapping, coordinate generation)
- **tcsh shell scripts**: Pipeline orchestration (`TOPSpipe.tcsh` is the main entry point)

## Build & tooling

Uses **Hatch** as the project manager and build backend (`hatchling` + `hatch-vcs` for version from git tags).

No `setup.py` or `requirements.txt` — all metadata lives in `pyproject.toml`.

## Commands

```sh
# Activate virtualenv
/home/tiira/.virtualenvs/radar-hail-probability/bin/activate


## Airflow Integration

This package is deployed as a containerized service in FMI's Airflow v2.11 radar production system. The integration pattern:

- **Deployment**: Docker container with this package installed (built via `Containerfile`, image `quay.io/fmi/radar-hail-probability:vx.y.z`).
- **Airflow tasks**: `@task.docker` decorator is used to invoke Python API.
- **No DAGs in this repo**: Workflow orchestration lives in the separate Airflow radar production repository
- **Robustness**: Handle missing/corrupted input files and edge cases gracefully, log processing steps

## Conventions
- **Python ≥ 3.12** — use modern syntax freely (e.g. `match`, `type` aliases, `X | Y` unions)

### Style
- Follow Black formatting
- Naming, comments, etc. in English
- Mention corresponding legacy names for key variables in comments/docstrings if helpful
- It's better to briefly quote legacy code than to refer to line numbers
- Use `logging` module for debug/info/warning/error messages
- Type hinting for all functions
- Succinct, to the point documentation
- Avoid repeating bad practices from legacy code

## Data flow (legacy)
1. Input: IRIS radar composites — 45 dBZ and 50 dBZ TOPS products in cartesian IRIS format
2. Model data: 0°C and −20°C isotherm heights from text files
3. Processing: C programs combine radar and model data to compute POH and LHI fields
4. Output: POH/LHI fields in IRIS format and GIF visualizations

## Key domain concepts
* **POH**: Probability Of Hail — primary output product
* **LHI**: Large Hail Index — secondary output product
* **HHI**: Holleman Hail Index, Modified POH using 0.1 probability units, no upper limit. Formula 10*POH
* **THI**: Tuovinen Hail Index, modified HHI
  * TOP <= 1200m, THI=HHI+2
  * 1200m < TOP <= 1700m, THI=HHI+1
  * 1700m < TOP <= 3500m, THI=HHI
  * TOP > 3500m, THI=HHI-1
* LHI, HHI and THI are defined for integer values:
  * HHI and THI >= 0
  * LHI may have negative values
* **TOPS products**: Radar echo top heights at given reflectivity thresholds (45/50 dBZ)
* **Isotherms**: 0°C and −20°C altitude levels from NWP model data, used as inputs to POH formula
* **undetect**: encodes pixels that have no physical TOPS value
* **nodata**: encodes pixels that are outside radar composite coverage


## Repository Structure

```
legacy/          # Original production C code and tcsh scripts (reference only)
legacy/data/     # example data files
src/             # python code
tests/           # pytest tests
```

## Development Context

- The Python rewrite is the active development target
- The legacy code in `legacy/` serves as reference for understanding the algorithms
- The legacy C code depends on external headers (`sigtypes.h`, `product.h`, etc.) from the IRIS/FMI software environment — these are not in the repo and the legacy code is not expected to compile standalone
