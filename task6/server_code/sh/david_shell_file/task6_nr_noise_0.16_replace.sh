#!/bin/sh
#SBATCH --account=stats
#SBATCH --job-name=task6_r_part2
#SBATCH -c 16
#SBATCH -t 5-00:00
#SBATCH -C mem192
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=sz3091@columbia.edu
 
module load anaconda
#Command to execute Python program
python BOMP_testing.py --config ./configs/no_replacement/task6_nr_noise_0.16_replace.yaml
#End of script