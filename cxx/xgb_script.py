import xgboost as xgb
import numpy as np
import pandas as pd

task='cv' # {'train','cv','predict'}
param = {'max_depth':4, 'eta':0.8, 'silent':1, 'objective':'reg:linear', 'tree_method':'auto'}
model_name='0001.model'

if task == 'train' or task == 'cv':
    print 'start to load training data ... '
    valid_data = np.loadtxt("valid.csv")
#    valid_data = valid_data[np.random.choice(valid_data.shape[0], 100000)]
    data=valid_data[:,0:(valid_data.shape[1]-1)]
    label=valid_data[:,valid_data.shape[1]-1]
    del valid_data

    print 'prepare training'
    dtrain = xgb.DMatrix(data, label=np.log(label+1), missing = -999.0)


if task == 'cv':
    print 'start cv'

    res = xgb.cv(param, dtrain, num_boost_round=200, nfold=5,
                 metrics={'rmse'}, seed = 0,
                 callbacks=[xgb.callback.print_evaluation(show_stdv=False),
                            xgb.callback.early_stop(3)])

if task == 'train':
    num_round = 200;
    bst = xgb.train(param, dtrain, num_boost_round=num_round, verbose_eval=True);

    bst.save_model(model_name);

if task == 'predict':
    test_data = np.loadtxt("test_feature.csv");    
    dtest = xgb.DMatrix(test_data, missing = -999.0)
    bst = xgb.Booster();
    bst.load_model(model_name)
    pred = bst.predict(dtest)
    pred[pred<0] = 0 # set positive
    pred = np.exp(pred)-1
    submission = pd.DataFrame({'id':np.arange(len(pred)), 'Demanda_uni_equil': pred})
    submission.to_csv('submit.csv', index=False)
