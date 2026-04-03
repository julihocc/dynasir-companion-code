# DynaSIR Companion Code

This repository contains the analysis scripts, Jupyter notebooks, and supporting code for the DynaSIR paper. It is extracted as a standalone module to provide researchers with a specialized workspace for running and reproducing our benchmark analyses and experiments.

## Repository Structure

- `notebooks/`: Contains the Jupyter notebooks for generating plots and analyzing simulation outputs (e.g., `report/report.ipynb`).
- `scripts/`: Contains Python scripts for executing benchmarks, diagnosing sensitivity, and verifying components such as dynamic vs. static populations.

## Usage

This project relies on the core `dynasir` library. To run the analysis locally:

1. Clone this repository (or initialize it as a submodule within the main `dynasir-paper` repository).
2. Ensure you have the required dependencies (refer to `pyproject.toml` in the main paper repo, or install them via `uv`).
3. Execute the notebooks via Jupyter or run the scripts directly from the command line.

## See Also

- **Paper Repository**: [julihocc/dynasir-paper](https://github.com/julihocc/dynasir-paper)
- **Library**: [julihocc/dynasir](https://github.com/julihocc/dynasir)
