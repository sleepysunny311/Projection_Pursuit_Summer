#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
#SBATCH --account=stats         # Replace ACCOUNT with your group account name
#SBATCH --job-name=sm_gs     # The job name.
#SBATCH -c 8                    # The number of cpu cores to use
#SBATCH -t 3-00:00                # Runtime in D-HH:MM
#SBATCH -C mem192         # The memory the job will use per cpu core
#SBATCH --mail-type=ALL	
#SBATCH --mail-user=ts3464@columbia.edu
 
module load anaconda

#Command to execute Python program
python BOMP_testing.py --config configs/bomp_1000_600_20_small.yaml
 
#End of script
