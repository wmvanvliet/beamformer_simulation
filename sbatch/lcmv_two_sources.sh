#!/bin/bash

# Make sure to request only the resources you really need to avoid cueing
#SBATCH -t 24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH -n 1

# A name for the job
#SBATCH --job-name lcmv_2s

# Do the analysis for each vertex.
# SBATCH --array=0-3756
#SBATCH --array=0446,0464,0462,0463,0441,0444

#SBATCH --output=lcmv_two_sources.out --open-mode=append

# Location to write the logfile to
LOG_FILE=logs/lcmv_two_sources.log

VERTEX_NUMBER=$(printf "%04d" $SLURM_ARRAY_TASK_ID)

# Load the python environment
module load anaconda3

# Tell BLAS to only use a single thread
export OMP_NUM_THREADS=1

# Run the script
srun python ../lcmv_two_sources.py -v $SLURM_ARRAY_TASK_ID -n 0.1 2>&1 | sed -e "s/^/$VERTEX_NUMBER:  /" >> $LOG_FILE
