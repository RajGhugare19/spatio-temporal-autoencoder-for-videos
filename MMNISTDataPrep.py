import os
import torch
import torch.nn as nn
import numpy as np


traindatapath = './MMNIST/train'
testdatapath = './MMNIST/test'

k = np.load('mnist_test_seq.npy') #Shape = (20, 10000, 64, 64)

for seq_id in range(9995):
    seq = torch.tensor(np.expand_dims(k[:,seq_id,:,:],axis=1))
    save_path = traindatapath + '/id-' + str(seq_id) + '.pt'
    torch.save(seq,save_path)

for seq_id in range(5):
    seq = torch.tensor(np.expand_dims(k[:,seq_id+9995,:,:],axis=1))
    save_path = testdatapath + '/id-' + str(seq_id) + '.pt'
    torch.save(seq,save_path)