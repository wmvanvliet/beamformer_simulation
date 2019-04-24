#!/bin/bash

# Make sure to request only the resources you really need to avoid cueing
#SBATCH -t 10:00
#SBATCH --mem-per-cpu=4G
#SBATCH -n 1

# Do the analysis for each subject. This should correspond with the SUBJECTS
# variable below.
#SBATCH --array=1-10

# Find the current subject
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID - 1]}

# Location to write the logfile to
LOG_FILE=logs/simulate_raw-$SLURM_ARRAY_TASK_ID.log

# Load the python environment
module load anaconda3

# Tell BLAS to only use a single thread
export OMP_NUM_THREADS=1

# Run the script
srun -o $LOG_FILE python ../simulate_raw.py -v $SLURM_ARRAY_TASK_ID -n 1
