
from ase import Atoms, Atom
from ase.calculators.vasp import Vasp
from ase.spacegroup import crystal
from ase.io import read, write
import numpy as np
import os
import time

### Function for creating job submitting file  ###
def bsub_file(run_name,runfile,run_command,np='8'):    # inputs are job name anf filename
    #print('qsub_file')
    pwd=os.getcwd()
    with open(runfile,'w') as run_file:
        run_file.write('#!/bin/sh\n')
        run_file.write('#SBATCH -o  out\n')
        run_file.write('#SBATCH -p  compute\n')
        run_file.write('#SBATCH -J %s\n' % run_name)
        run_file.write('#SBATCH -t 240:00:00\n')
        run_file.write('#SBATCH -N 1\n')
        #run_file.write('#SBATCH -n '+np+' \n')
        run_file.write('#SBATCH -n 32 \n')
        run_file.write('#SBATCH --mem=60GB\n')
        run_file.write('\n export OMP_NUM_THREADS=1 \n')
        run_file.write('cd %s \n' % pwd)
        run_file.write(run_command)

### Funtion for running job  ###
def bsub_run(run_name,runfile):
    run='sbatch -J '+run_name+' < '+runfile+' > run_id.txt '   # runining command and saving job ID to run_id.txt
#    print(run)
    os.system(run)   # submitting job

### Function checking job's status by given name and ID  ###

def bsub_stat(run_name):
    with open('run_id.txt','r') as idfile:
        id_line=idfile.readline()
    run_id=id_line.split(' ')[-1][:-1]
    
    run_bjobs='squeue > bjobs.txt'
    os.system(run_bjobs)
    with open('bjobs.txt','r') as bjobs_file:
        bjobs_line=bjobs_file.readlines()
    run_bjobs='Finished' # if not found it's mean that job is finished
    for i in bjobs_line:
        if (run_id in i):
            #print(i)
            run_bjobs= i.split()[4]  # getting status of the run
    
    return run_bjobs

### Function for waintg until given job is Completed ###
def bsub_finished(run_name,run_bjobs):
    if run_bjobs != "Finished":     #  if the run was, found do while circle until it complete
        j=0
        while run_bjobs!='Finished':
            #print(run_qstat)
            time.sleep(5)
            j=j+1
            #print(j)
            run_bjobs=bsub_stat(run_name)   # Updating job's status
        print(j)
    return run_bjobs



