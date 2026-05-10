#!/bin/bash -l
#SBATCH -J plot_1_10
#SBATCH -p priority
#SBATCH -t 480:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 4G
#SBATCH -o slurm-%j.out


export OMP_NUM_THREADS=1

REPO_DIR="/mnt/dftevn/home/hayk/workdir/2D/W_N/WN_codes"
cd "${REPO_DIR}"

module purge
conda activate wn_env


python src/dftkit/analysis/plot_dos_pdos_band.py \
  --input-db data/processed/wn_materials.db \
  --ids 1,2,3,4,5,6,7,8,9,10 \
  --calc-name DOS_calc \
  --output-root data/calculations \
  --plots-subdir plot_2\
  --emin -8 \
  --emax 8
