# kaggle-inventory

## Some important points

1. one-period-ahead forecast != two-period-ahead forecast
2. blur of history => parameter risk
3. use public leaderboard scores to validate the two effects!

## run cross validation
```
./run_ffm.sh
./run_knn.sh

cp client.csv client_ro.csv

./run_ffm.s.sh

./run_aggregate.sh

./run_xgb.sh
```
