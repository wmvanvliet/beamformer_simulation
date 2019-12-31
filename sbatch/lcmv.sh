#!/bin/bash

# Make sure to request only the resources you really need to avoid cueing
#SBATCH -t 15:00
#SBATCH --mem-per-cpu=2G
#SBATCH -n 1

# A name for the job
#SBATCH --job-name lcmv

# Do the analysis for each vertex.
#SBATCH --array=0-4637

# Location to write the logfile to
LOG_FILE=logs/lcmv-$SLURM_ARRAY_TASK_ID.log

# Load the python environment
module load mesa
module load anaconda3

# Tell BLAS to only use a single thread
export OMP_NUM_THREADS=1

# Start a virtual framebuffer to render things to
Xvfb :99 -screen 0 1400x900x24 -ac +extension GLX +render -noreset &
export DISPLAY=:99.0

# Run the script
srun -o $LOG_FILE python ../lcmv.py -v $SLURM_ARRAY_TASK_ID -n 0.1
