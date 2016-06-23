import xgboost as xgb
import numpy as np

print 'start to load data ... '
valid_data = np.loadtxt("valid.csv")
data=valid_data[:,0:(valid_data.shape[1]-1)]
label=valid_data[:,valid_data.shape[1]-1]
del valid_data

print 'done; prepare training'
dtrain = xgb.DMatrix(data, label=np.log(label+1), missing = -999.0)

print 'done; start cv'
param = {'max_depth':2, 'eta':0.05, 'silent':1, 'objective':'reg:linear'}

res = xgb.cv(param, dtrain, num_boost_round=100, nfold=5,
             metrics={'rmse'}, seed = 0,
             callbacks=[xgb.callback.print_evaluation(show_stdv=False),
                        xgb.callback.early_stop(3)])


