#!/bin/bash

#SBATCH --array=0-14
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=24:00:00
#SBATCH --job-name=pl_nc_nc_thresh
#SBATCH --mail-type=END,FAIL

module load gcc/8.2.0 python/3.10.4 
source ../../loss-of-plasticity/bin/activate
python online_expr_nc_threshold.py -c temp_cfg/$SLURM_ARRAY_TASK_ID.json