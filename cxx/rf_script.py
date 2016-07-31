from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

task='validate' # {'train','validate','predict'}
is_final=False
model=0

if model==0:
    model_name='0000.rf'
    select=np.concatenate((np.arange(5,23), np.arange(25,30))) 
    #select=np.arange(5, 30)
elif model==1:
    model_name='0001.rf'
    select=np.arange(5, 26) #model1: without ffm features
elif model==2:
    model_name='0002.rf'
    select=np.arange(18,31) #model2: (almost) without history orders
elif model==3:
    model_name='0003.rf'
    select=[17, 19, 20, 21, 22, 23, 24, 25, 28] #model3: no client identities
else:
    model_name='raw_id.rf'
    select=np.arange(0, 5) # use the raw id as features

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
    return data, np.log(label+1)

if task == 'train' or task == 'validate':
    dX71, dy71=get_data('valid71_cache/valid.bin', 10382849)
    dX81, dy81=get_data('valid81_cache/valid.bin', 10406868)
    dX91, dy91=get_data('valid91_cache/valid.bin', 10408713)

if task == 'train' or task == 'validate':
    rf=RandomForestRegressor(n_estimators=100, n_jobs=24, min_samples_leaf=100, random_state=0, verbose=1)
    if task == 'train' and is_final:        
        rf.fit(dX91, dy91)
    elif task == 'train' and not is_final:
        rf.fit(dX81, dy81)
        dy91_pred=rf.predict(dX91)
        print np.sqrt(np.mean((dy91_pred - dy91)**2))
    elif task == 'validate':
        rf.fit(dX81, dy81)
        dy91_pred=rf.predict(dX91)
        print np.sqrt(np.mean((dy91_pred - dy91)**2))

if task == 'train':
    if is_final:
        test_data = np.fromfile("test0_cache/test_feature.bin", dtype=np.float32);    
    else:
        test_data = np.fromfile("test1_cache/test_feature.bin", dtype=np.float32);
    test_data = np.reshape(test_data, (6999251, len(test_data)/6999251));
    if select is not None:
        test_data = test_data[:, select]
    pred = rf.predict(test_data)
    pred[pred<0] = 0 # set positive
    pred = np.exp(pred)-1
    submission = pd.DataFrame({'id':np.arange(len(pred)), 'Demanda_uni_equil': pred})
    submission.to_csv('submit.csv', index=False)
