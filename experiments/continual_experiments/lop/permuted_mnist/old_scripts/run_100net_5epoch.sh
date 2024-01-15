#!/bin/bash

#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=16384
#SBATCH --time=48:00:00
#SBATCH --job-name=pl_nc_100net_5epoch
#SBATCH --output=slurm/pl_nc_100net_5epoch.out
#SBATCH --mail-type=END,FAIL

module load gcc/8.2.0 python_gpu/3.10.4
source ../../loss-of-plasticity/bin/activate
python multi_param_expr.py -c cfg/bp/100net_5epoch.json
python online_expr.py -c temp_cfg/0.json