import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from MMNISTDataLoader import MMNISTTestDataset,MMNISTTrainDataset
from custom_conv_lstm import STAE
device = torch.device("cuda:0")
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 2}

max_epochs = 100

training_set = MMNISTTrainDataset(9995)
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = MMNISTTestDataset(5)
validation_generator = torch.utils.data.DataLoader(validation_set)
model = STAE(1).to(device)

for epoch in range(max_epochs):
    
    for local_batch in training_generator:
        
        local_batch = local_batch.to(device)/255.0
        local_output = model(local_batch)
        loss = model.criterion(local_batch,local_output)
        model.zero_grad()
        loss.backward()
        model.optimizer.step()
        model.zero_grad()
        print(loss.item())

    for local_batch in validation_generator:
        local_batch = local_batch.to(device)/255.0
        local_output = model(local_batch)
        loss = model.criterion(local_batch,local_output).item()
        print("EPOCH = ",epoch)
        print("loss = ",loss)