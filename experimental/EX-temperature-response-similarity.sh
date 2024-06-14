#!/bin/bash
# TODO:        Before running the script, there are 3 fields that should
#              change with every job submission:
#              1. job name
#              2. output file
#              3. parameters passed to driver_experiments
#

# SBATCH --job-name=orgTemp
# SBATCH --mail-user=jpic@umich.edu
# SBATCH --mail-type=BEGIN,END
# SBATCH --nodes=1
# SBATCH --cpus-per-task=36
# SBATCH --mem=180G
# SBATCH --time=24:00:00
# SBATCH --account=indikar99
# SBATCH --partition=standard

python EX-temperature-response-similarity.py
