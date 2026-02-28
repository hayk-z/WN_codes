#!/bin/bash
#SBATCH -J vasp_ysu2
#SBATCH -p compute
#SBATCH -t 240:00:00
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --mem 120G
#SBATCH -o slurm-%j.out

export OMP_NUM_THREADS=1
cd /mnt/dftevn/home/hayk/workdir/2D/W_N/WN_codes
module purge
module load vasp/6.5.1_gnu_ompi_mkl_omp
mpirun -np 32 vasp_std
