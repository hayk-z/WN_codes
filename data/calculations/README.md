# Calculations Data

This directory stores machine-readable records of DFT calculations.

## Layout
- `jobs/`: input/job metadata before submission
- `results/`: parsed output summaries after completion
- `logs/`: optional run logs and scheduler outputs
- `calculation_registry.csv`: one-line registry for quick filtering and audit

## Recommended workflow
1. Add a job file in `jobs/`.
2. Run the calculation on your compute resource.
3. Parse outputs and save a compact summary to `results/`.
4. Append or update `calculation_registry.csv`.
