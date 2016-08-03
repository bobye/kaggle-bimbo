#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -t 24:00:00
#SBATCH --mem=100gb



export PYTHONPATH=$PYTHONPATH:/home/jxy198/xgboost/python-package
module load gsl

PHOME=/home/jxy198/kaggle-inventory

echo "Start KNN!"

#cd $PHOME/cxx/valid71_cache
#python $PHOME/cxx/knn_script.py &
cd $PHOME/cxx/valid81_cache
python $PHOME/cxx/knn_script.py	&
cd $PHOME/cxx/valid91_cache
python $PHOME/cxx/knn_script.py	&
cd $PHOME/cxx/test1_cache
python $PHOME/cxx/knn_script.py	&

wait
