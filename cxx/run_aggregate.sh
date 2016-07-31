#!/bin/bash

#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH -t 24:00:00
#SBATCH --mem=16gb



export PYTHONPATH=$PYTHONPATH:/home/jxy198/xgboost/python-package
module load gsl
PHOME=/home/jxy198/kaggle-inventory

offset=1

echo "Start Feature Eng!"
time {
#cd $PHOME/cxx/valid7${offset}_cache
#$PHOME/cxx/main 7${offset} rrr &

cd $PHOME/cxx/valid8${offset}_cache
$PHOME/cxx/main 8${offset} rrr &

cd $PHOME/cxx/valid9${offset}_cache
$PHOME/cxx/main 9${offset} rrr &

cd $PHOME/cxx/test${offset}_cache
$PHOME/cxx/main t${offset} rrr &

wait
}
