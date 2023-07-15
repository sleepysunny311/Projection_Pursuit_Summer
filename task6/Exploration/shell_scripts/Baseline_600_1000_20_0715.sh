#!/bin/sh
#SBATCH --account=stats
#SBATCH --job-name=BOMP_600_1000_20_0715
#SBATCH -c 16
#SBATCH -t 5-00:00
#SBATCH -C mem192
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=sz3091@columbia.edu

module load anaconda
#Command to execute Python program
python OMP_testing.py --config-name BOMP_600_1000_20_r_0715.yaml --config-path
#End of script
