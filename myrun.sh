#!/bin/bash -l
#SBATCH -J zpe
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

python src/dftkit/workflows/zpe_gibbs_workflow.py   --input-db data/processed/h_adsorption_materials.db   --config configs/zpe_calc.yaml   --slurm-config configs/slurm_aznavour.conf   --calc-name ZPE_Gibbs_calc   --ids 1-4, 7-30

