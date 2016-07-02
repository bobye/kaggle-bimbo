#!/bin/bash

#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH -t 24:00:00
#SBATCH --mem=16gb



export PYTHONPATH=$PYTHONPATH:/home/jxy198/xgboost/python-package
module load gsl
PHOME=/home/jxy198/kaggle-inventory

echo "Start Feature Eng!"
time {
cd $PHOME/cxx/valid81_cache
$PHOME/cxx/main 81 rrr &

cd $PHOME/cxx/valid91_cache
$PHOME/cxx/main 91 rrr &

cd $PHOME/cxx/test1_cache
$PHOME/cxx/main t1 rrr &

wait
}
