import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Reshape 
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint


Xtrain=np.fromfile('eenn_tr.index.bin', dtype=np.int32)
ytrain=np.fromfile('eenn_tr.label.bin', dtype=np.float32)
Xtest=np.fromfile('eenn_te.index.bin', dtype=np.int32)
ytest=np.fromfile('eenn_te.label.bin', dtype=np.float32)


size_of_index=np.max(Xtrain) + 1
Xtrain=np.reshape(Xtrain, (len(Xtrain)/4, 4))
Xtest=np.reshape(Xtest, (len(Xtest)/4, 4))

test_filter = np.any(Xtest < size_of_index, axis=1)
Xvalid=Xtest[test_filter,:]
yvalid=ytest[test_filter]

embedding_dim=10
channel_num=4

nn_ee = Sequential()
nn_ee.add(Embedding(size_of_index, embedding_dim, input_length=channel_num))
nn_ee.add(Reshape(target_shape=(embedding_dim*channel_num,)))
nn_ee.add(Dense(1000, init='uniform'))
nn_ee.add(Activation('relu'))
nn_ee.add(Dropout(0.5))
nn_ee.add(Dense(200, init='uniform'))
nn_ee.add(Activation('relu'))
nn_ee.add(Dropout(0.5))
nn_ee.add(Dense(1))
nn_ee.compile(loss='mean_squared_error', optimizer='sgd')



nn_ee.fit(Xtrain, ytrain, 
          validation_data = (Xvalid, yvalid), 
          nb_epoch=1, batch_size=1024)

nn_ee.save_weights('nn_k10_sel.txt')

