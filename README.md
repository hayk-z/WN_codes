# W-N Materials DFT Project

This repository stores W-N 2D materials data and Python workflows for database building, VASP input preparation, job submission, and DOS/PDOS/Band step orchestration.

## Project tree

```text
.
├── configs/
│   ├── dft_defaults.yaml
│   ├── dos_calc_hse.yaml
│   ├── dos_calc_pbe.conf
│   ├── dos_calc_pbe.yaml
│   ├── myrun.sh
│   └── slurm_ysu2.conf
├── data/
│   ├── calculations/
│   ├── processed/
│   │   ├── wn_materials.db
│   │   ├── wn_materials_export.json
│   │   └── wn_materials_export.yaml
│   └── raw/
├── scripts/
├── src/
│   └── dftkit/
│       ├── analysis/
│       │   └── plot_dos_pdos_band.py
│       ├── db/
│       │   └── query_wn_db.py
│       ├── io/
│       │   └── build_wn_materials_db.py
│       ├── utils/
│       │   ├── bsub_funcs_vasp.py
│       │   ├── calc_funcs_vasp.py
│       │   └── create_myrun_from_config.py
│       └── workflows/
│           ├── dos_pdos_band_hse_workflow.py
│           ├── dos_pdos_band_workflow.py
│           ├── prepare_vasp_inputs.py
│           └── submit_prepared_jobs.py
├── tests/
└── environment.yml
```

## Environment

```bash
conda env update --file environment.yml --prune
conda activate wn_env
```

### `scripts/volcano_plot_n.py`

Purpose:
- Generate volcano plot (`log(i0)` vs `Delta G_H*`) for W-N 2D materials.
- Read `Reports_gen/Table_1.csv` by default.
- Compute Gibbs free energy and exchange current, then save:
  - `Reports_gen/Adsorption_gibbs_with_i0.csv`
  - `Reports_gen/volcano_plot.png`
  - `Reports_gen/volcano_plot.pdf`

Marker style:
- Default marker for all points: circle.
- Only `W2N3` points are shape-mapped by lattice symmetry:
  - `square` lattice -> square marker
  - `hexagonal` lattice -> hexagon marker

Defaults:
- Labels: hidden by default.
- Legend: hidden by default.

Options:
- `--input` (default: `Reports_gen/Table_1.csv`)
- `--show-labels` / `--hide-labels`
- `--show-legend` / `--hide-legend`

Example:
```bash
python3 scripts/volcano_plot_n.py
```

## Source Files

### `src/dftkit/io/build_wn_materials_db.py`

Purpose:
- Build ASE database from raw adsorption export files.

Inputs:
- CSV table (default: `data/raw/adsorption_materials_export/Adsorption_gibbs_with_i0.csv`)
- Material folders with `material_database_entry.json` and `FINAL_STRUCTURE.vasp`

Options:
- `--csv`
- `--materials-root`
- `--output-db`
- `--expected-count`

Examples:
```bash
python src/dftkit/io/build_wn_materials_db.py
```
```bash
python src/dftkit/io/build_wn_materials_db.py \
  --csv data/raw/adsorption_materials_export/Adsorption_gibbs_with_i0.csv \
  --materials-root data/raw/adsorption_materials_export/materials \
  --output-db data/processed/wn_materials.db \
  --expected-count 9
```

### `src/dftkit/db/query_wn_db.py`

Purpose:
- Query ASE DB and optionally export matched records to JSON/YAML.

Inputs:
- ASE database (`--db`)

Options:
- `--db`
- `--material`
- `--crystal-system`
- `--limit`
- `--json-out`
- `--yaml-out`
- `--export-all`

Examples:
```bash
python src/dftkit/db/query_wn_db.py --db data/processed/wn_materials.db
```
```bash
python src/dftkit/db/query_wn_db.py \
  --db data/processed/wn_materials.db \
  --material USPEX_VARIABLE_W-N-2D_2971 \
  --limit 5
```
```bash
python src/dftkit/db/query_wn_db.py \
  --db data/processed/wn_materials.db \
  --export-all \
  --json-out data/processed/wn_materials_export.json \
  --yaml-out data/processed/wn_materials_export.yaml
```

### `src/dftkit/workflows/prepare_vasp_inputs.py`

Purpose:
- Prepare per-material VASP job folders (`INCAR`, `KPOINTS`, `POSCAR`, `POTCAR`, `myrun.sh`) from DB/JSON/YAML records.

Inputs:
- Dataset file: `.db`, `.json`, `.yml`, `.yaml` via `--input-db`
- Calculation label via `--calc-name`

Options:
- `--input-db` (required)
- `--calc-name` (required)
- `--ids` (optional subset)
- `--dft-config` (default: `configs/dft_defaults.yaml`)
- `--slurm-config` (default: `configs/slurm_ysu2.conf`)
- `--output-root` (default: `data/calculations`)
- `--potcar` / `--potcar-root` (default: `data/potcars`; path to `potpaw_PBE` or its parent)

Examples:
```bash
python src/dftkit/workflows/prepare_vasp_inputs.py \
  --input-db data/processed/wn_materials.db \
  --calc-name wn_relax_all
```
```bash
python src/dftkit/workflows/prepare_vasp_inputs.py \
  --input-db data/processed/wn_materials.db \
  --calc-name wn_relax_selected \
  --ids 1,2,5
```

### `src/dftkit/workflows/submit_prepared_jobs.py`

Purpose:
- Submit already prepared folders that contain `myrun.sh`.

Inputs:
- Prepared root directory with folders like `id_<id>_<material>/`

Options:
- `--prepared-root` (required)
- `--ids` (optional subset)
- `--script-name` (default: `myrun.sh`)
- `--submit-cmd` (default: `sbatch`)
- `--dry-run`

Examples:
```bash
python src/dftkit/workflows/submit_prepared_jobs.py \
  --prepared-root data/calculations/wn_materials_relax_all
```
```bash
python src/dftkit/workflows/submit_prepared_jobs.py \
  --prepared-root data/calculations/wn_materials_relax_all \
  --ids 1,3,5 \
  --dry-run
```

### `src/dftkit/workflows/dos_pdos_band_workflow.py`

Purpose:
- Multi-step VASP workflow driver with SLURM submission and monitoring.
- Step layout:
  - `01_relax`: geometry relaxation
  - `02_scf`: static SCF from `01_relax/CONTCAR`
  - `03_dos`: DOS run from step 2 with restart files (`CHGCAR`, `WAVECAR`)
  - `04_band`: 2D band workflow:
    - identifies 2D symmetry/layer-group,
    - writes summary/JSON outputs,
    - generates canonical 2D band path KPOINTS,
    - prepares/submits band run.
- Runs sequentially from `--start-step` to step 4.

Inputs:
- Structure source via `--input-db` (`.db` or `.json`)
- Workflow YAML config (default: `configs/dos_calc_pbe.yaml`)
- SLURM config (default: `configs/slurm_ysu2.conf`)
- Optional ID subset

Options:
- `--input-db` (required)
- `--calc-name` (default: `DOS_calc`)
- `--ids`
- `--config` (workflow YAML)
- `--slurm-config`
- `--output-root`
- `--potcar` / `--potcar-root` (default: `data/potcars`)
- `--start-step` (`1|2|3|4`)
- `--poll-seconds`
- `--max-wait-hours`
- `--dry-run`

Key `step4_band` YAML parameters:
- `symmetry_tolerance`
- `angle_tolerance`
- `aperiodic_dir` (`auto`, `0`, `1`, `2`)
- `band_points_per_segment`
- `vasp_tags` (same static tags as DOS style, without `nedos`)

Examples:
```bash
python src/dftkit/workflows/dos_pdos_band_workflow.py \
  --input-db data/processed/wn_materials.db \
  --ids 2 \
  --config configs/dos_calc_pbe.yaml \
  --slurm-config configs/slurm_ysu2.conf \
  --calc-name DOS_calc
```
```bash
python src/dftkit/workflows/dos_pdos_band_workflow.py \
  --input-db data/processed/wn_materials.db \
  --ids 2 \
  --config configs/dos_calc_pbe.yaml \
  --start-step 3
```
```bash
python src/dftkit/workflows/dos_pdos_band_workflow.py \
  --input-db data/processed/wn_materials.db \
  --ids 2 \
  --config configs/dos_calc_pbe.yaml \
  --start-step 4
```
```bash
python src/dftkit/workflows/dos_pdos_band_workflow.py \
  --input-db data/processed/wn_materials.db \
  --ids 2 \
  --config configs/dos_calc_pbe.yaml \
  --start-step 4 \
  --dry-run
```

### `src/dftkit/workflows/dos_pdos_band_hse_workflow.py`

Purpose:
- HSE version of DOS/PDOS/Band workflow with the same step logic and sequential execution from `--start-step` to step 4.
- Uses the same SLURM config (`configs/slurm_ysu2.conf`) and HSE settings from `configs/dos_calc_hse.yaml`.

HSE step logic:
- `01_relax`: same as relax, but `LWAVE=.TRUE.`, `LCHARG=.TRUE.`
- `02_scf`: HSE SCF (`LHFCALC`, `HFSCREEN`, `AEXX`, `ALGO=D`, `TIME`) with restart from step 1 (`CHGCAR`, `WAVECAR`)
- `03_dos`: HSE DOS/PDOS (`ISMEAR=-5`, `NEDOS=2000`, `LORBIT=11`) with restart from step 2
- `04_band`: HSE band (no `NEDOS`) with 2D k-path generation as in PBE workflow

Inputs:
- Structure source via `--input-db` (`.db` or `.json`)
- Workflow YAML config (default: `configs/dos_calc_hse.yaml`)
- SLURM config (default: `configs/slurm_ysu2.conf`)
- Optional ID subset

Options:
- `--input-db` (required)
- `--calc-name` (default: `DOS_HSE_calc`)
- `--ids`
- `--config` (workflow YAML)
- `--slurm-config`
- `--output-root`
- `--potcar` / `--potcar-root` (default: `data/potcars`)
- `--start-step` (`1|2|3|4`)
- `--poll-seconds`
- `--max-wait-hours`
- `--dry-run`

Examples:
```bash
python src/dftkit/workflows/dos_pdos_band_hse_workflow.py \
  --input-db data/processed/wn_materials.db \
  --ids 2 \
  --config configs/dos_calc_hse.yaml \
  --slurm-config configs/slurm_ysu2.conf \
  --calc-name DOS_HSE_calc \
  --start-step 1
```
```bash
python src/dftkit/workflows/dos_pdos_band_hse_workflow.py \
  --input-db data/processed/wn_materials.db \
  --ids 2 \
  --config configs/dos_calc_hse.yaml \
  --start-step 2 \
  --dry-run
```

### `src/dftkit/workflows/zpe_gibbs_workflow.py`

Purpose:
- Run vibrational calculations for H adsorption systems and apply ZPE-based Gibbs correction.
- Computes `ZPE(H2)` once (if missing), computes `ZPE(H*)` from 3 adsorbed-H modes, then updates DB/report.
- For H* inputs, writes POSCAR with selective dynamics: slab atoms (non-H) fixed (`F F F`), H atoms movable (`T T T`).
- If fewer than 3 positive real H* modes are found, the workflow uses the top-3 modes by `|frequency|` (with warning), so execution does not stop.

Correction used:
- `delta_zpe_ev = zpe_h_star_ev - 0.5 * zpe_h2_ev`
- `delta_g_h_ev = adsorption_energy_ev + delta_zpe_ev - t_delta_s_ev`

Inputs:
- Structure source via `--input-db` (`.db` or `.json`; DB updates are applied only for `.db`)
- Workflow YAML config (default: `configs/zpe_calc.yaml`)
- SLURM config (default: `configs/slurm_ysu2.conf`)
- Optional ID subset

Options:
- `--input-db` (required)
- `--calc-name` (default: `ZPE_Gibbs_calc`)
- `--ids` (optional subset; supports single IDs, comma lists, and ranges like `1-4 6-20` or `1,3,5-8`)
- `--config` (workflow YAML)
- `--slurm-config`
- `--output-root`
- `--potcar` / `--potcar-root` (default from config or `data/potcars`)
- `--poll-seconds`
- `--max-wait-hours`
- `--dry-run`

Main outputs:
- `<calc_root>/general_inputs.json` (stores `zpe_h2_ev`, `t_delta_s_ev`)
- `<calc_root>/zpe_results.json` (full per-ID frequency and correction details)
- `<calc_root>/zpe_gibbs_report.csv` (summary table: `id`, `name`, `composition`, `adsorption_site_type`, `delta_zpe_ev`, `zpe_h_star_ev`, `half_zpe_h2_ev`, `delta_g_h_ev`)

Examples:
```bash
python src/dftkit/workflows/zpe_gibbs_workflow.py \
  --input-db data/processed/h_adsorption_materials.db \
  --config configs/zpe_calc.yaml \
  --slurm-config configs/slurm_aznavour.conf \
  --calc-name ZPE_Gibbs_calc \
  --ids 1-4 7-20
```
```bash
python src/dftkit/workflows/zpe_gibbs_workflow.py \
  --input-db data/processed/h_adsorption_materials.db \
  --config configs/zpe_calc.yaml \
  --calc-name ZPE_Gibbs_calc \
  --ids 18,22-25 \
  --dry-run
```

### `src/dftkit/analysis/plot_dos_pdos_band.py`

Purpose:
- Plot DOS, PDOS, and band-structure figures from completed workflow folders.

Expected folders per ID:
- DOS files in `03_dos/vasprun.xml`
- Band files in `04_band/vasprun.xml` and `04_band/KPOINTS`

Generated plots (`<id_dir>/plots/`):
- `dos_total.png`
- `pdos_element.png`
- `band_structure.png`
- `band_structure_projected.png` (partial/projected band)
- `band_dos_combined.png` (band on left + vertical DOS panel with total DOS and element PDOS)

Options:
- `--input-db` (required)
- `--calc-name` (default: `DOS_calc`)
- `--ids` (optional subset)
- `--output-root` (default: `data/calculations`)
- `--dos-step` (default: `03_dos`)
- `--band-step` (default: `04_band`)
- `--plots-subdir` (default: `plots`)
- `--emin` (default: `-6`)
- `--emax` (default: `6`)

Examples:
```bash
python src/dftkit/analysis/plot_dos_pdos_band.py \
  --input-db data/processed/wn_materials.db \
  --calc-name DOS_calc \
  --ids 2 \
  --output-root data/calculations \
  --emin -6 \
  --emax 6
```

### `src/dftkit/utils/create_myrun_from_config.py`

Purpose:
- Generate standalone `myrun.sh` from cluster `.conf` file.

Inputs:
- Cluster config file (`--config`)

Options:
- `--config` (default: `configs/slurm_ysu2.conf`)
- `--output` (default: `myrun.sh`)
- `--workdir` (default: current directory)

Examples:
```bash
python src/dftkit/utils/create_myrun_from_config.py \
  --config configs/slurm_ysu2.conf \
  --output myrun.sh \
  --workdir /path/to/calculation_dir
```

### `src/dftkit/utils/calc_funcs_vasp.py`

Purpose:
- Utility functions for VASP setup and output parsing.
- Used by workflows to write VASP inputs (`set_vasp`) and post-process outputs.

Inputs/options/examples:
- Library module (no CLI arguments).
- Used internally from workflow scripts.

### `src/dftkit/utils/bsub_funcs_vasp.py`

Purpose:
- Utility functions for SLURM submission/status helper calls (`bsub_run`, `bsub_stat`, etc.).

Inputs/options/examples:
- Library module (no CLI arguments).
- Used internally from workflow scripts.
