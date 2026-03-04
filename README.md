# W-N Materials DFT Project

This repository organizes W-N 2D materials data, analysis scripts, and an ASE database workflow.

## Project structure

```text
.
├── configs/
│   └── dft_defaults.yaml                # default DFT settings template
├── data/
│   ├── raw/
│   │   └── adsorption_materials_export/ # source CSV + material JSON/VASP files
│   ├── processed/
│   │   ├── wn_materials.db              # ASE database (generated)
│   │   ├── wn_materials_export.json     # JSON export (generated)
│   │   └── wn_materials_export.yaml     # YAML export (generated)
│   └── calculations/                    # calculation tracking scaffolding
├── reports/
│   ├── figures/
│   └── tables/
├── scripts/                             # plotting / one-off scripts
├── src/
│   └── dftkit/
│       ├── io/
│       │   └── build_wn_materials_db.py # builds ASE DB from raw inputs
│       └── db/
│           └── query_wn_db.py           # query and export DB content
├── tests/
└── environment.yml
```

## Environment

The Conda environment name is `wn_env` (see `environment.yml`).

Activate once:
```bash
conda activate wn_env
```
Then run python commands normally.

## SLURM submission config (YSU2)

Cluster submission parameters are now separated in config files under `configs/`.

- `configs/slurm_ysu2.conf`: YSU2 defaults (job name, partition, walltime, memory, modules, run command)
- `configs/myrun.sh`: reads a cluster config, creates a submit script, and calls `sbatch`

Usage from repository root:

```bash
bash configs/myrun.sh
```

Optional arguments:

```bash
bash configs/myrun.sh configs/slurm_ysu2.conf /path/to/calculation_dir
```

To add another cluster, copy `configs/slurm_ysu2.conf`, adjust values, and pass that file to `configs/myrun.sh`.

## Prepare VASP input folders (Python)

Use `prepare_vasp_inputs.py` to generate VASP-ready job directories with:
- `INCAR`, `KPOINTS`, `POSCAR`, `POTCAR`, `myrun.sh`

The script reads:
- DFT defaults from `configs/dft_defaults.yaml`
- Cluster run settings from `configs/slurm_ysu2.conf`
- Structures from `.db`, `.json`, or `.yml/.yaml`

Run from repository root with `wn_env`:

```bash
/mnt/dftevn/home/hayk/miniconda3/envs/wn_env/bin/python \
src/dftkit/workflows/prepare_vasp_inputs.py \
--input-db data/processed/wn_materials.db \
--calc-name wn_relax_all
```

Generate only specific IDs:

```bash
/mnt/dftevn/home/hayk/miniconda3/envs/wn_env/bin/python \
src/dftkit/workflows/prepare_vasp_inputs.py \
--input-db data/processed/wn_materials.db \
--calc-name wn_relax_selected \
--ids 1,2,5
```

Use JSON/YAML exported records instead of ASE DB:

```bash
/mnt/dftevn/home/hayk/miniconda3/envs/wn_env/bin/python \
src/dftkit/workflows/prepare_vasp_inputs.py \
--input-db data/processed/wn_materials_export.json \
--calc-name wn_relax_json
```

Custom options:
- `--dft-config` path to DFT defaults YAML
- `--slurm-config` path to cluster config
- `--output-root` output base folder (default: `data/calculations`)
- `--potcar-root` POTCAR family path (default: `/mnt/dftevn/opt/vasp/pseudo/potpaw_PBE`)

Output layout:

```text
data/calculations/<input_name>_<calc_name>/
  id_<id>_<material>/
    INCAR
    KPOINTS
    POSCAR
    POTCAR
    myrun.sh
```

## Generate only myrun.sh from cluster config (Python)

If needed, generate just a `myrun.sh` template from config:

```bash
python src/dftkit/utils/create_myrun_from_config.py \
  --config configs/slurm_ysu2.conf \
  --output myrun.sh \
  --workdir /path/to/calculation_dir
```

## Submit prepared jobs by DB IDs (Python)

Use `submit_prepared_jobs.py` to submit `myrun.sh` inside prepared folders:

```bash
python src/dftkit/workflows/submit_prepared_jobs.py \
  --prepared-root data/calculations/wn_materials_relax_all \
  --ids 1,3,5
```

Submit all prepared IDs:

```bash
python src/dftkit/workflows/submit_prepared_jobs.py \
  --prepared-root data/calculations/wn_materials_relax_all
```

Dry-run (no submission, only print commands):

```bash
python src/dftkit/workflows/submit_prepared_jobs.py \
  --prepared-root data/calculations/wn_materials_relax_all \
  --ids 1,3 \
  --dry-run
```

## Build the ASE database

From repository root:

```bash
python src/dftkit/io/build_wn_materials_db.py
```

This reads:
- `data/raw/adsorption_materials_export/Adsorption_gibbs_with_i0.csv`
- `data/raw/adsorption_materials_export/materials/*/material_database_entry.json`
- `data/raw/adsorption_materials_export/materials/*/FINAL_STRUCTURE.vasp`

And writes:
- `data/processed/wn_materials.db`

## Query the DB

Print matched records:

```bash
python src/dftkit/db/query_wn_db.py --db data/processed/wn_materials.db
```

Filter examples:

```bash
python src/dftkit/db/query_wn_db.py \
  --db data/processed/wn_materials.db \
  --crystal-system monoclinic \
  --limit 3
```

```bash
python src/dftkit/db/query_wn_db.py \
  --db data/processed/wn_materials.db \
  --material USPEX_VARIABLE_W-N-2D_2971
```

## Export DB to JSON and YAML

```bash
python src/dftkit/db/query_wn_db.py \
  --db data/processed/wn_materials.db \
  --export-all \
  --json-out data/processed/wn_materials_export.json \
  --yaml-out data/processed/wn_materials_export.yaml
```

## Typical workflow

1. Update raw data in `data/raw/adsorption_materials_export/`.
2. Rebuild DB with `build_wn_materials_db.py`.
3. Run query/export with `query_wn_db.py`.
4. Use exported JSON/YAML or direct ASE DB access for analysis and plotting.
