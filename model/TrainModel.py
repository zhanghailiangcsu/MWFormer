# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 09:36:30 2023

@author: Administrator
"""
import numpy as np
import torch
from model.Model import MyDataSet
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
#import matplotlib.pyplot as plt
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler

def dataset_sep(mz_list,intensity_list,weights,val_size):
    '''
    Randomly split the dataset
    '''
    seed=1
    np.random.seed(1)
    n = len(weights)
    perm = np.random.permutation(n)
    n_train = int(n*(1-val_size))
    perm_train = perm[0:n_train]
    perm_val = perm[n_train:]
    mz_list_train = [mz_list[x] for x in perm_train] 
    mz_list_val = [mz_list[x] for x in perm_val] 
    intensity_list_train = [intensity_list[x] for x in perm_train]
    intensity_list_val = [intensity_list[x] for x in perm_val] 
    weights_train = [weights[x] for x in perm_train]
    weights_val = [weights[x] for x in perm_val]
    return mz_list_train,intensity_list_train,weights_train,mz_list_val,intensity_list_val,weights_val

def Train(model,mz_list_train,intensity_list_train,weights_train,batch_size,lr,epochs):
    '''
    Training the WeightFormer model model,train_loss,val_loss=Train(model, mz_list_train, intensity_list_train, weights_train, batch_size, lr, epochs)
    '''
    mz_list_train,intensity_list_train,weights_train,mz_list_val,intensity_list_val,weights_val = dataset_sep(mz_list_train,intensity_list_train,weights_train,0.1)
    dataset_train = MyDataSet(mz_list_train,intensity_list_train,weights_train)
    dataloader_train = Data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_val = MyDataSet(mz_list_val,intensity_list_val,weights_val)
    dataloader_val = Data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr,weight_decay=0.01)
    steps = epochs*len(dataloader_train)
    scheduler = CosineLRScheduler(optimizer, t_initial=steps, lr_min=0.1 * lr, warmup_t=int(0.1*steps), warmup_lr_init=0)
    step_count = 0
    device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    train_loss = []
    val_loss = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = []
        for step,(mz_,intensity_,weights_) in enumerate(dataloader_train):
            mz_ = mz_.to(device)
            intensity_ = intensity_.to(device)
            intensity_ = intensity_.unsqueeze(2).float()
            weights_ = weights_.to(device).float()
            weights_ = weights_.unsqueeze(1)
            out = model(mz_,intensity_)
            loss = criterion(out,weights_)
            epoch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(step_count)
            step_count += 1
        train_loss.append(epoch_loss)
        print(str(epoch)+'epoch train_loss'+str(np.nanmean(epoch_loss)))
        
        model.eval()
        with torch.no_grad():
            val_step_loss = []
            for step,(mz_,intensity_,weights_) in enumerate(dataloader_val):
                mz_ = mz_.to(device)
                intensity_ = intensity_.to(device)
                intensity_ = intensity_.unsqueeze(2).float()
                weights_ = weights_.to(device).float()
                weights_ = weights_.unsqueeze(1)
                out = model(mz_,intensity_)
                loss = criterion(out,weights_)
                val_step_loss.append(loss.item())
            val_loss.append(val_step_loss)
            print(str(epoch)+'epoch val_loss'+str(np.nanmean(val_step_loss)))
            # torch.save(model.state_dict(),'model.pkl')
    return model,train_loss,val_loss

def Predict(model,mz_list_test,intensity_list_test,weights_test,batch_size):
    '''
    Using a trained model to predict the molecular weight of the spectrum
    '''
    dataset_test = MyDataSet(mz_list_test,intensity_list_test,weights_test)
    dataloader_test = Data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    true_weights = []
    predict_weights = []
    model.eval()
    with torch.no_grad():
        for step,(mz_,intensity_,weights_) in tqdm(enumerate(dataloader_test)):
            mz_ = mz_.to(device)
            intensity_ = intensity_.to(device)
            intensity_ = intensity_.unsqueeze(2).float()
            weights_ = weights_.to(device).float()
            weights_ = weights_.unsqueeze(1)
            out = model(mz_,intensity_)
            true_weights.append(weights_.detach().cpu().numpy())
            predict_weights.append(out.detach().cpu().numpy())
    true_weights = [i.ravel() for i in true_weights]
    true_weights = np.concatenate(true_weights)
    predict_weights = [i.ravel() for i in predict_weights]
    predict_weights = np.concatenate(predict_weights)
    return true_weights,predict_weights

def PlotLoss(train_loss,val_loss):
    '''
    Draw loss curve
    '''
    train_loss2 = [np.nanmean(i) for i in train_loss]
    val_loss2 = [np.nanmean(i) for i in val_loss]
    plt.plot(train_loss2,label='Train')
    plt.plot(val_loss2,label='Val')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
