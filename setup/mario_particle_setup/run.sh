#!/bin/bash -x
#SBATCH --account=jhpc54
#SBATCH --nodes=4
#SBATCH --job-name=mario_val_lb_lpt
#SBATCH --ntasks-per-node=128
#SBATCH --time=24:00:00
#SBATCH --partition=dc-cpu
#SBATCH --output=slurm-out.%j
#SBATCH --error=slurm-err.%j

source env_maia

mclean
srun maia_nasal properties_particles.toml 
#srun maia_nasal properties_run.toml 
