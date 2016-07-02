#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -t 24:00:00
#SBATCH --mem=100gb

export OMP_NUM_THREADS=24

export PYTHONPATH=$PYTHONPATH:/home/jxy198/xgboost/python-package
module load gsl
PHOME=/home/jxy198/kaggle-inventory

echo "Start Learning!"
cd $PHOME/cxx
python xgb_script.py
