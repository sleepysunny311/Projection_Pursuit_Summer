#!/bin/sh
#
#
#SBATCH --account=ACCOUNT       # Replace ACCOUNT with your group account name
#SBATCH --job-name=Projection_Pursuit_Task5
#SBATCH --cpus-per-task=1       # The number of cpu cores to use
#SBATCH --time=10:00:00            # Runtime in D-HH:MM
#SBATCH --mem=16gb              # The memory the job will use per cpu core
 
module load anaconda
 
#Command to execute Python program
python example.py
 
#End of script