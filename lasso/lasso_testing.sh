#!/bin/sh
#SBATCH --account=stats
#SBATCH --job-name=BOMP_300_500_10_0721
#SBATCH -c 16
#SBATCH -t 5-00:00
#SBATCH -C mem192
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=ts3464@columbia.edu

module load anaconda
#Command to execute Python program
python lasso_testing.py --config lasso_testing.yaml 
#End of script
