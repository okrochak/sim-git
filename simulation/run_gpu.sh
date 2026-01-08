#!/bin/bash -x
#SBATCH --account=jhpc54
#SBATCH --nodes=1
#SBATCH --job-name=cyclic-simulation
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --partition=dc-gpu-devel
#SBATCH --output=slurm-out.%j
#SBATCH --error=slurm-err.%j
#SBATCH --gres=gpu:4

source env_maia_gpu

mclean
srun maia properties_run.toml 
