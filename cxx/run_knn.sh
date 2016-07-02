#!/bin/bash

#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH -t 24:00:00
#SBATCH --mem=32gb



export PYTHONPATH=$PYTHONPATH:/home/jxy198/xgboost/python-package
module load gsl

PHOME=/home/jxy198/kaggle-inventory

cd $PHOME/cxx/test1_cache

echo "Start KNN!"
python $PHOME/cxx/knn_script.py
