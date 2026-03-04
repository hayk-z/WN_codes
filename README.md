# W-N Materials DFT Project

This repository stores W-N 2D materials data and Python workflows for database building, VASP input preparation, job submission, and DOS/PDOS/Band step orchestration.

## Project tree

```text
.
├── configs/
│   ├── dft_defaults.yaml
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
│       ├── db/
│       │   └── query_wn_db.py
│       ├── io/
│       │   └── build_wn_materials_db.py
│       ├── utils/
│       │   ├── bsub_funcs_vasp.py
│       │   ├── calc_funcs_vasp.py
│       │   └── create_myrun_from_config.py
│       └── workflows/
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
- `--potcar-root` (default: `/mnt/dftevn/opt/vasp/pseudo/potpaw_PBE`)

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
- DOS/PDOS/Band workflow driver.
- Current implemented stage: `01_relax` (prepare + submit + monitor).
- Creates step folders: `01_relax`, `02_dos`, `03_pdos`, `04_band`.

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
- `--poll-seconds`
- `--max-wait-hours`
- `--dry-run`

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
  --ids 1,2 \
  --config configs/dos_calc_pbe.yaml \
  --dry-run
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
