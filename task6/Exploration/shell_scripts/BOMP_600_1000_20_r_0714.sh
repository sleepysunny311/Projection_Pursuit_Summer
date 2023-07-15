#!/bin/sh
#SBATCH --account=stats
#SBATCH --job-name=BOMP_600_1000_20_0714
#SBATCH -c 16
#SBATCH -t 5-00:00
#SBATCH -C mem192
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=sz3091@columbia.edu

module load anaconda
#Command to execute Python program
python BOMP_testing.py --config-name ./configs/BOMP_600_1000_20_r_0714.yaml
#End of script
