# kaggle-inventory

## Some important points

1. one-period-ahead forecast != two-period-ahead forecast
2. blur of history => parameter risk
3. overfit validation period => high model complexity

Use public leaderboard scores wisely to observe those effects and make scientific judgements!

## How to run
```
./run_ffm.sh
./run_knn.sh

cp client.csv client_ro.csv

./run_ffm.s.sh

./run_aggregate.sh

./run_xgb.sh
```
