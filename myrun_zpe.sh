#!/bin/bash
#SBATCH -J zpe_gibbs
#SBATCH -p priority
#SBATCH -t 480:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 4G
#SBATCH -o slurm-%j.out


export OMP_NUM_THREADS=1
cd /mnt/dftevn/home/hayk/workdir/2D/W_N/WN_codes

module purge
conda activate wn_env

python src/dftkit/workflows/zpe_gibbs_workflow.py \
  --input-db data/processed/h_adsorption_materials.db \
  --config configs/zpe_calc.yaml \
  --slurm-config configs/slurm_ysu2.conf \
  --calc-name ZPE_Gibbs_calc \
  --ids 5-8
