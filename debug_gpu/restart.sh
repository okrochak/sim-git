#!/bin/bash -x
#SBATCH --account=slfse
#SBATCH --nodes=4 #8
#SBATCH --job-name=SIM
#SBATCH --ntasks-per-node=100 #128
#SBATCH --time=01:00:00
#SBATCH --partition=dc-cpu
#SBATCH --output=slurm-out.%j
#SBATCH --error=slurm-err.%j

source env_maia

mclean
srun maia properties_restart.toml 

