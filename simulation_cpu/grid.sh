#!/bin/bash -x
#SBATCH --account=jhpc54
#SBATCH --nodes=4 #4
#SBATCH --job-name=nasal-cavity
#SBATCH --ntasks-per-node=10 #128
#SBATCH --time=00:15:00
#SBATCH --partition=dc-cpu
#SBATCH --output=slurm-out.%j
#SBATCH --error=slurm-err.%j

source env_maia

mclean
srun maia properties_grid.toml 
cp out/grid.Netcdf restart/