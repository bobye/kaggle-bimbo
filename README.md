# kaggle-inventory

This is the C++ codebase I created during participating the Kaggle Competition: Grupo Bimbo Inventory Demand. The code is not cleaned, so blindly to compile and run will not work anyway. And don't expect you can run the code. Someone asked me how can one do feature engineering in C++. This is what I did by heavily using the C++ Standard Library, especially the unordered_map() API (Please see `./cxx/main.cc` file). 

BTW. Don't judge my engineering capability by looking at this code :) I have compiled a Chinese [post](https://zhuanlan.zhihu.com/p/22266330) on this competition.

## Some important points

1. one-period-ahead forecast != two-period-ahead forecast
2. blur of history => parameter risk
3. overfit validation period => high model complexity

Use public leaderboard scores wisely to observe those effects and make scientific judgements!

## How to run
```
./run_ffm.sh
./run_knn.sh
./ffm2eenn tr
./ffm2eenn te
./run_eenn.sh

cp client.csv client_ro.csv

./run_ffm.s.sh

./run_aggregate.sh

./run_xgb.sh
```
