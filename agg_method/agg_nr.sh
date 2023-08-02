#!/bin/sh
#
#SBATCH --account=stats         # Replace ACCOUNT with your group account name
#SBATCH --job-name=gs0910     # The job name.
#SBATCH -c 16                    # The number of cpu cores to use
#SBATCH -t 5-00:00                # Runtime in D-HH:MM
#SBATCH -C mem192         # The memory the job will use per cpu core
#SBATCH --mail-type=ALL	
#SBATCH --mail-user=ts3464@columbia.edu
 
module load anaconda

#Command to execute Python program
python agg_testing.py --config configs/bmp_gs_300_600_20_test.yaml
 
#End of script
