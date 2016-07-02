#!/bin/bash

#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH -t 24:00:00
#SBATCH --mem=16gb
######PBS -l nodes=1:ppn=16:walltime=24:00:00:mem=64gb



export PYTHONPATH=$PYTHONPATH:/home/jxy198/xgboost/python-package
module load gsl
module load gnutools

PHOME=/home/jxy198/kaggle-inventory

echo "Start Feature Eng!"
cd $PHOME/cxx/valid81_cache
cp client.csv client_ro.csv
$PHOME/cxx/main 81 rwr

k=2
echo "Start FFM k=$k"
$PHOME/libffm-regression/ffm-train -t 20 -r 0.0027 -k $k --auto-stop -p ffm_te2.s.txt ffm_tr.s.txt ffm_k${k}_sel.s.txt
$PHOME/libffm-regression/ffm-predict ffm_te.s.txt ffm_k${k}_sel.s.txt ffm_te_pred.s.txt > /tmp/null
