#!/bin/bash
#SBATCH -J dos_pbe
#SBATCH -p priority
#SBATCH -t 480:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 4G

export OMP_NUM_THREADS=1
cd /mnt/dftevn/home/hayk/workdir/2D/W_N/WN_codes

module purge
conda activate wn_env


python src/dftkit/workflows/dos_pdos_band_workflow.py   --input-db data/processed/wn_materials.db   --ids 2, 10   --config configs/dos_calc_pbe.yaml  --start-step 1 --slurm-config configs/slurm_ysu2.conf   --calc-name DOS_calc


"myrun_h.sh" 18L, 533B                                                                                                      
