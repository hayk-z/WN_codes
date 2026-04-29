#!/bin/bash -l
#SBATCH -J h1_dos
#SBATCH -p compute
#SBATCH -t 480:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 4G
#SBATCH -o slurm-%j.out

export OMP_NUM_THREADS=1
cd /scratch/users/hayk_zakaryan/workdir/2D/WN/WN_codes
module purge
conda activate wn_env


python src/dftkit/workflows/dos_pdos_band_workflow.py   --input-db data/processed/h_adsorption_materials.db   --ids 55, 65, 29, 36   --config configs/dos_calc_pbe.yaml  --start-step 3 --slurm-config configs/slurm_aznavour.conf   --calc-name DOS_h1_calc_test


