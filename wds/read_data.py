import numpy as np
import scipy.io as scio
from scipy import sparse as sp

train_data = sp.load_npz('./data_sparse.npz')
train_label = np.load('./20ng_label.npy')
train_label = train_label.reshape(len(train_label),1)
train_data = train_data.toarray()
np.random.seed(2018)
np.random.shuffle(train_data)
np.random.seed(2018)
np.random.shuffle(train_label)
train_label = train_label[:,0]
train_data = train_data.T
voc = np.load('./vocab.npy')
train_data = np.array(train_data,order = 'C')
print(train_data.shape,train_label.shape,voc.shape)
