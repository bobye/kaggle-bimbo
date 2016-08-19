import xgboost as xgb
import numpy as np
import pandas as pd

task='validate' # {'train','validate','predict'}
is_final=False 
num_round=90+1
model=1
#param = {'max_depth':4, 'eta':0.05, 'silent':1, 'objective':'reg:linear', 'tree_method':'exact', 'nthread':24}
param = {'max_depth':8, 'eta':0.05, 'gamma':10, 'silent':1, 'objective':'reg:linear', 'tree_method':'exact', 'nthread':24}

if model==0:
    model_name='0000.model'
    #select=np.concatenate((np.arange(5,23), np.arange(25,30))) 
    select=np.arange(5, 30)
    #select=np.concatenate((np.arange(5,26), [28, 29, 32, 34]))
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
    ## add noise to dim 24
    # addnoise=data[:,24] != -999
    # noise=np.random.normal(0,1,np.sum(addnoise))
    # data[addnoise,24] += noise
    label=valid_data[:,valid_data.shape[1]-1]
    if select is not None:
        data=data[:,select]
    del valid_data

    print 'prepare training'
    dtrain = xgb.DMatrix(data, label=np.log(label+1), weight=weights, missing = -999.0)
    return dtrain

if task == 'train' or task == 'validate':
    #dtrain71=get_data('valid71_cache/valid.bin', 10382849)
    dtrain81=get_data('valid81_cache/valid.bin', 10406868)
    dtrain91=get_data('valid91_cache/valid.bin', 10408713)

if task == 'train' or task == 'validate':
    if task == 'train' and is_final:
        watchlist=[(dtrain91, 'train')]
	#watchlist=[(dtrain90, 'train')]
        bst = xgb.train(param, dtrain91, 
			num_boost_round = num_round, verbose_eval=True,
			evals=watchlist)
    elif task == 'train' and not is_final:
        watchlist=[(dtrain81, 'train'), (dtrain91, 'eval')]
	#watchlist=[(dtrain80, 'train'), (dtrain90, 'eval')]
        bst = xgb.train(param, dtrain81,
                        num_boost_round = num_round, verbose_eval=True,
                        evals=watchlist, early_stopping_rounds=1)
    elif task == 'validate':
        watchlist=[(dtrain81, 'train'), (dtrain91, 'eval')]
	#watchlist=[(dtrain80, 'train'), (dtrain90, 'eval')]
        bst = xgb.train(param, dtrain81,
                        num_boost_round = 1000, verbose_eval=True,
                        evals=watchlist, early_stopping_rounds=1)
    print bst.get_fscore()
    bst.save_model(model_name)
    bst.dump_model('xgb.dump', with_stats=True)

if task == 'predict' or task == 'train':
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
