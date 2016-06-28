from sklearn.neighbors import NearestNeighbors
import numpy as np


k=3
d=2

raw_data = np.loadtxt('ffm_tr_knn_data.txt')
train = raw_data[:,0:(raw_data.shape[1]-1)]
label = raw_data[:,(raw_data.shape[1]-1)]
del raw_data
nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(train)

raw_data_txt = np.loadtxt('ffm_te_knn_data.txt', delimiter='\n', dtype=str);
raw_data = -999 * np.ones((len(raw_data_txt), d+1))
for count,line in enumerate(raw_data_txt):
    tmp = np.array([float(x) for x in line.split(' ')])
    if len(tmp)==d+1:
        raw_data[count,:] = tmp;
del raw_data_txt

test = raw_data[:,0:(raw_data.shape[1]-1)]
test_label= raw_data[:,(raw_data.shape[1]-1)]
del raw_data

size=test.shape[0];
distances, indices = nbrs.kneighbors(test)
knn_est = np.mean(np.reshape(label[indices.ravel()], indices.shape), 1)
knn_est[test_label == -999] = -999
distances = np.mean(distances, 1)
distances[test_label == -999] = -999
print "rmse =", np.sqrt(np.mean((test_label - knn_est)[test_label!=-999]**2))
np.savetxt('knn_te_pred.txt', np.concatenate((knn_est,distances)).reshape((2,size)).T)
