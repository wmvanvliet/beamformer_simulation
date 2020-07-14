#!/bin/bash

# Make sure to request only the resources you really need to avoid cueing
#SBATCH -t 2:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH -n 1

# A name for the job
#SBATCH --job-name lcmv_megset

# Do the analysis for each subject
#SBATCH --array=1,2,4,5,6,7

#SBATCH --output=lcmv_megset.out --open-mode=append

# Location to write the logfile to
LOG_FILE=logs/lcmv_megset_${SLURM_ARRAY_TASK_ID}.log

# Load the python environment
module load anaconda

# Tell BLAS to only use a single thread
export OMP_NUM_THREADS=1

# Run the script
srun python ../lcmv_megset.py -s $SLURM_ARRAY_TASK_ID > $LOG_FILE
