import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.cross_validation import LabelKFold

task='train' # {'train','cv','predict'}
is_final=False
has_history=False
#param = {'max_depth':4, 'eta':0.8, 'silent':1, 'objective':'reg:linear', 'tree_method':'exact', 'nthread':8}
param = {'max_depth':5, 'eta':0.8, 'silent':1, 'objective':'reg:linear', 'tree_method':'auto', 'nthread':8}
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
    label_kfold = LabelKFold(cv_folds, max(cv_folds)+1)
    res = xgb.cv(param, dtrain, num_boost_round=300, folds=label_kfold,
                 seed = 0,
                 callbacks=[xgb.callback.print_evaluation(show_stdv=False),
                            xgb.callback.early_stop(3)])

if task == 'train':
    num_round = 300;
    cv_folds = np.loadtxt("folds.txt")
    if is_final:
        bst = xgb.train(param, dtrain, num_boost_round=num_round, verbose_eval=True);
    else:
        train0=dtrain.slice(np.nonzero([x!=0 for x in cv_folds]))
        valid0=dtrain.slice(np.nonzero([x==0 for x in cv_folds]))        
        watchlist=[(train0, 'train'), (valid0, 'eval')]
        bst = xgb.train(param, train0,
                        num_boost_round = num_round, verbose_eval=True,
                        evals=watchlist, early_stopping_rounds=3)
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
