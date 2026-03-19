#!/bin/bash -l
#SBATCH -J hse
#SBATCH -p compute
#SBATCH -t 240:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 4G
#SBATCH -o slurm-%j.out

export OMP_NUM_THREADS=1
cd /home/hayk_zakaryan/workdir/2D/WN/WN_codes 
module purge
conda activate wn_env

python src/dftkit/workflows/dos_pdos_band_hse_workflow.py  --input-db data/processed/wn_materials.db  --config configs/dos_calc_hse.yaml  --slurm-config configs/slurm_aznavour.conf  --calc-name DOS_HSE_calc  --start-step 1


