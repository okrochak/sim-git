#!/bin/bash -x
#SBATCH --account=jhpc54 #jhpc54
#SBATCH --nodes=1
#SBATCH --job-name=riken_unsteady
#SBATCH --ntasks-per-node=48
#SBATCH --time=24:00:00
#SBATCH --partition=dc-cpu
#SBATCH --output=slurm-out.%j
#SBATCH --error=slurm-err.%j

source env_maia
source ~/jureca-python/bin/activate
python ../process_point_data.py --input out/ --output probes.dat
