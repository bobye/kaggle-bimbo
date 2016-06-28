import xgboost as xgb
import numpy as np
import pandas as pd

task='cv' # {'train','cv','predict'}
has_history=True
#param = {'max_depth':4, 'eta':0.8, 'silent':1, 'objective':'reg:linear', 'tree_method':'exact', 'nthread':24}
param = {'max_depth':5, 'eta':0.8, 'silent':1, 'objective':'reg:linear', 'tree_method':'auto', 'nthread':24}
model_name='0003.model'

print param
# def myobj(preds, dtrain):
#     labels = dtrain.get_label()
#     preds = np.maximum(preds, 0)
#     grad = (preds - labels)
#     hess = np.ones(preds.shape)
#     hess[preds == 0] = 0
#     return grad, hess

# def myerror(preds, dtrain):
#     labels = dtrain.get_label()
#     preds = np.maximum(preds, 0)
#     return 'error', np.sqrt(((preds - labels) ** 2).mean())


if task == 'train' or task == 'cv':
    print 'start to load training data ... '
    valid_data = np.fromfile("valid.bin", dtype=np.float32)
    valid_data = np.reshape(valid_data, (10408713, len(valid_data)/10408713))
    if has_history:
        valid_data=valid_data[valid_data[:,0]!=-999]
    ## down-sample data for local run
    # valid_data = valid_data[np.random.choice(valid_data.shape[0], 100000)] 
    data=valid_data[:,0:(valid_data.shape[1]-1)]
    label=valid_data[:,valid_data.shape[1]-1]
    del valid_data

    print 'prepare training'
    dtrain = xgb.DMatrix(data, label=np.log(label+1), missing = -999.0)


if task == 'cv':
    print 'start cv'
    cv_folds = np.loadtxt("folds.txt")
    res = xgb.cv(param, dtrain, num_boost_round=300, folds=cv_folds,
                 seed = 0,
                 callbacks=[xgb.callback.print_evaluation(show_stdv=False),
                            xgb.callback.early_stop(3)])

if task == 'train':
    num_round = 200;
    bst = xgb.train(param, dtrain, num_boost_round=num_round, verbose_eval=True);
    print bst.get_fscore()
    bst.save_model(model_name);

if task == 'predict':
    test_data = np.fromfile("test_feature.bin", dtype=np.float32);    
    test_data = np.reshape(test_data, (6999251, len(test_data)/6999251));
    dtest = xgb.DMatrix(test_data, missing = -999.0)
    bst = xgb.Booster(param);
    bst.load_model(model_name)
    pred = bst.predict(dtest)
    pred[pred<0] = 0 # set positive
    pred = np.exp(pred)-1
    submission = pd.DataFrame({'id':np.arange(len(pred)), 'Demanda_uni_equil': pred})
    submission.to_csv('submit.csv', index=False)
