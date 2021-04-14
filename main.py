import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from UCSDDataLoader import UCSDValDataset,UCSDTrainDataset
from custom_conv_lstm import STAE

import wandb
wandb.login()

config = dict(
        learning_rate=0.0001,
        train_params = {'batch_size': 4,
                 'shuffle': True,
                 'num_workers': 2},
        val_params = {
            'batch_size': 1,
            'shuffle': True,
            'num_workers': 1
        },
        max_epochs = 100,
        input_channels = 1,
        device='cuda:0',
        architecture = "STAE",        
)

if torch.cuda.is_available() and config['device']=='cuda:0':
    
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True


def make(config):
    
    device = torch.device(config['device'])
    train_params = config['train_params']
    val_params = config['val_params']

    model = STAE(config).to(device)
    
    training_set = UCSDTrainDataset(2039)
    training_generator = torch.utils.data.DataLoader(training_set, **train_params)

    validation_set = UCSDValDataset(90)
    validation_generator = torch.utils.data.DataLoader(validation_set, **val_params)
    
    return model, training_generator, validation_generator

def model_pipeline(config):
    
    with wandb.init(project="STAE-UCSD-demo",config=config):
        config=wandb.config
        model,training_generator,validation_generator = make(config)
        model = train(model,training_generator,validation_generator,config)
        #test(model,config)

    return model

def train_log(log_values,epoch):
    wandb.log(log_values, step=epoch)

def save_model(model,chk_pnt,best_model = False):
    
    if best_model == False:
        save_path = 'saved_models/' + str(chk_pnt) + '.pt'
    else:
        save_path = 'saved_models/best.pt'
    
    torch.save(model,save_path)

def train(model,training_generator,validation_generator,config):
    
    max_epochs = config['max_epochs']
    device = torch.device(config['device'])
    log_values = dict(
        train_epochloss=0,
        val_epochloss=0,
    )
    least_loss_value = np.inf
    best_model = model
    for epoch in range(max_epochs):
        i = 0
        log_values['train_epochloss'] = 0   
        log_values['val_epochloss']
        
        for local_batch in training_generator:
        
            local_batch = local_batch.to(device)
            local_output = model(local_batch.float())
            loss = model.criterion(local_batch,local_output)
            model.zero_grad()
            loss.backward()
            model.optimizer.step()
            model.zero_grad()
            print(loss.item())
            log_values['train_epochloss'] += loss.item()
            
        
        for local_batch in validation_generator:
            local_batch = local_batch.to(device)
            local_output = model(local_batch.float())
            loss = model.criterion(local_batch,local_output).item()
            print("val loss = ", loss)
            log_values['val_epochloss'] += loss           

        train_log(log_values, epoch)
        save_model(save_model, chk_pnt=epoch)

        if log_values['val_epochloss'] <= least_loss_value:
            best_model = model
            least_loss_value = log_values['val_epochloss']

    save_model(save_model, chk_pnt=None, best_model=True)

final_model = model_pipeline(config)