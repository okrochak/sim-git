#!/bin/bash -x
#SBATCH --account=jhpc54
#SBATCH --nodes=20
#SBATCH --job-name=coarse_cycle_VN
#SBATCH --ntasks-per-node=128
#SBATCH --time=06:00:00
#SBATCH --partition=dc-cpu
#SBATCH --output=slurm-out.%j
#SBATCH --error=slurm-err.%j

source env_maia

mclean
srun maia_nasal properties_run.toml 
