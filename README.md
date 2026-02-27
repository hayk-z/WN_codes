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
