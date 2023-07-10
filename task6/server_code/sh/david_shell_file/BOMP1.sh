#!/bin/sh
#
#
#SBATCH --account=stats         # Replace ACCOUNT with your group account name
#SBATCH --job-name=task6_nr_part2     # The job name.
#SBATCH -c 16                    # The number of cpu cores to use
#SBATCH -t 5-00:00                # Runtime in D-HH:MM
#SBATCH -C mem192         # The memory the job will use per cpu core
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=sz3091@columbia.edu
 
module load anaconda

#Command to execute Python program
python BOMP_testing.py --config configs/no_replacement/task6_nr_part2.yaml
 
#End of script