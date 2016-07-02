import xgboost as xgb
import numpy as np
import pandas as pd

task='train' # {'train','validate','predict'}
is_final=False
num_round=112
model=0
#param = {'max_depth':4, 'eta':0.1, 'silent':1, 'objective':'reg:linear', 'tree_method':'exact', 'nthread':24}
param = {'max_depth':10, 'eta':0.05, 'silent':1, 'objective':'reg:linear', 'tree_method':'exact', 'nthread':24}

if model==0:
    model_name='0000.model'
    select=np.arange(5, 30)
elif model==1:
    model_name='0001.model'
    select=np.arange(5, 26) #model1: without ffm features
elif model==2:
    model_name='0002.model'
    select=np.arange(18,30) #model2: (almost) without history orders
elif model==3:
    model_name='0003.model'
    select=[17, 19, 20, 21, 22, 23, 24, 25, 28] #model3: no client identities
else:
    model_name='raw_id.model'
    select=np.arange(0, 5) # use the raw id as features

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

def get_data(filename, size, has_history = None, reweight = None):
    "read training data from .bin file"
    print 'start to load training data ... '
    valid_data = np.fromfile(filename, dtype=np.float32)
    valid_data = np.reshape(valid_data, (size, len(valid_data)/size))
    if has_history:
        valid_data=valid_data[valid_data[:,5]!=-999]
    weights = None
    if reweight:
        weights = np.ones(size)
        weights[valid_data[:,5]==-999] = reweight
    ## down-sample data for local run
    # valid_data = valid_data[np.random.choice(valid_data.shape[0], 100000)] 
    data=valid_data[:,0:(valid_data.shape[1]-1)]
    label=valid_data[:,valid_data.shape[1]-1]
    if select is not None:
        data=data[:,select]
    del valid_data

    print 'prepare training'
    dtrain = xgb.DMatrix(data, label=np.log(label+1), weight=weights, missing = -999.0)
    return dtrain

if task == 'train' or task == 'validate':
    dtrain=get_data('valid81_cache/valid.bin', 10406868)
    dvalid=get_data('valid91_cache/valid.bin', 10408713)

if task == 'train' or task == 'validate':
    if task == 'train':
        watchlist=[(dvalid, 'train'), (dtrain, 'eval')]
        bst = xgb.train(param, dvalid, 
			num_boost_round=num_round, verbose_eval=True,
			evals=watchlist)
    elif task == 'validate':
        watchlist=[(dtrain, 'train'), (dvalid, 'eval')]
        bst = xgb.train(param, dtrain,
                        num_boost_round = 1000, verbose_eval=True,
                        evals=watchlist, early_stopping_rounds=3)
    print bst.get_fscore()
    pred = bst.predict(dvalid)
    label = dvalid.get_label()
    np.savetxt('errors.txt', np.concatenate((pred, label-pred)).reshape((2,len(label))).T, fmt='%.5f')
    bst.save_model(model_name)
    bst.dump_model('xgb.dump', with_stats=True)

if (task == 'train' and is_final) or (task == 'predict'):
    if is_final:
        test_data = np.fromfile("test0_cache/test_feature.bin", dtype=np.float32);    
    else:
        test_data = np.fromfile("test1_cache/test_feature.bin", dtype=np.float32);
    test_data = np.reshape(test_data, (6999251, len(test_data)/6999251));
    if select is not None:
        test_data = test_data[:, select]
    dtest = xgb.DMatrix(test_data, missing = -999.0)
    bst = xgb.Booster(param);
    bst.load_model(model_name)
    pred = bst.predict(dtest)
    pred[pred<0] = 0 # set positive
    pred = np.exp(pred)-1
    submission = pd.DataFrame({'id':np.arange(len(pred)), 'Demanda_uni_equil': pred})
    submission.to_csv('submit.csv', index=False)
