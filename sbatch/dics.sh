#!/bin/bash

# Make sure to request only the resources you really need to avoid cueing
#SBATCH -t 3:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH -n 1

# A name for the job
#SBATCH --job-name dics

# Do the analysis for each vertex.
#SBATCH --array=0-3756

#SBATCH --output=dics.out --open-mode=append

# Location to write the logfile to
LOG_FILE=logs/dics.log

VERTEX_NUMBER=$(printf "%04d" $SLURM_ARRAY_TASK_ID)

# Load the python environment
module load anaconda

# Tell BLAS to only use a single thread
export OMP_NUM_THREADS=1

# Run the script
srun python ../dics.py -v $SLURM_ARRAY_TASK_ID -n 0.1 2>&1 | sed -e "s/^/$VERTEX_NUMBER:  /" >> $LOG_FILE
