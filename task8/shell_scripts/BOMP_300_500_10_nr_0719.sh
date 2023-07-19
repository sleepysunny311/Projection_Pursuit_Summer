#!/bin/sh
#SBATCH --account=stats
<<<<<<< HEAD:task8/shell_scripts/Baseline_300_500_1020_0719.sh
#SBATCH --job-name=BOMP_300_500_1020_0719
=======
#SBATCH --job-name=BOMP_300_500_10_0719
>>>>>>> 7cb372dfe58074ad1fbac9e669fe4ac939d9a946:task8/shell_scripts/BOMP_300_500_10_nr_0719.sh
#SBATCH -c 16
#SBATCH -t 5-00:00
#SBATCH -C mem192
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=sz3091@columbia.edu

module load anaconda
#Command to execute Python program
<<<<<<< HEAD:task8/shell_scripts/Baseline_300_500_1020_0719.sh
python OMP_testing.py --config-name BOMP_300_500_1020_nr_0719.yaml --config-path ./configs
=======
python BOMP_testing.py --config-name BOMP_300_500_10_nr_0719.yaml --config-path ./configs
>>>>>>> 7cb372dfe58074ad1fbac9e669fe4ac939d9a946:task8/shell_scripts/BOMP_300_500_10_nr_0719.sh
#End of script
