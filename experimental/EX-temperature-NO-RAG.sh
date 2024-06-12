#!/bin/bash
# TODO:        Before running the script, there are 3 fields that should
#              change with every job submission:
#              1. job name
#              2. output file
#              3. parameters passed to driver_experiments
# 

#SBATCH --job-name=tempLlama
#SBATCH --mail-user=jpic@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --account=indikar99
#SBATCH --partition=standard
#SBATCH --array=0-20

python EX-temperature-NO-RAG.py $SLURM_ARRAY_TASK_ID