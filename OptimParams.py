# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:26:05 2023

@author: Administrator
"""
import optuna
from Model import WeightFormer, MyDataSet
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import torch
from timm.scheduler import CosineLRScheduler
import pickle
from utils import CalWeights
from TrainModel import Predict

def objective(trial):
    '''
    Define objective function
    '''
    hidden = trial.suggest_categorical('hidden',[32,64,128,256])
    batch_size = trial.suggest_categorical('batch_size',[16,32,64])
    n_layers = trial.suggest_categorical('n_layers',[3,4,5,6,7])
    attn_heads = trial.suggest_categorical('attn_heads',[2,4,8,16])
    epochs = trial.suggest_int('epochs',5,30)
    lr = trial.suggest_float('lr',1e-5,1e-3, log=True)
    
    dataset_train = MyDataSet(mz_list_train,intensity_list_train,weights_train)
    dataloader_train = Data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WeightFormer(1000,hidden, n_layers, attn_heads, 0)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr,weight_decay=0.01)
    steps = epochs*len(dataloader_train)
    scheduler = CosineLRScheduler(optimizer, t_initial=steps, lr_min=0.1 * lr, warmup_t=int(0.1*steps), warmup_lr_init=0)
    step_count = 0
    
    for epoch in range(epochs):
        model.train()
        for step,(mz_,intensity_,weights_) in enumerate(dataloader_train):
            mz_ = mz_.to(device)
            intensity_ = intensity_.to(device)
            intensity_ = intensity_.unsqueeze(2).float()
            weights_ = weights_.to(device).float()
            weights_ = weights_.unsqueeze(1)
            out = model(mz_,intensity_)
            loss = criterion(out,weights_)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(step_count)
            step_count += 1
    model.eval()
    with torch.no_grad():
        true_weights,predict_weights = Predict(model,mz_list_test,intensity_list_test,weights_test,batch_size)
        rmse,_ = CalWeights(true_weights,predict_weights)
    return rmse

if __name__ == '__main__':
    with open('E:/github/WeightFormer/EIMSdata/mz_list_train.pickle','rb') as f:
        mz_list_train = pickle.load(f)
    with open('E:/github/WeightFormer/EIMSdata/intensity_list_train.pickle','rb') as f:
        intensity_list_train = pickle.load(f)
    with open('E:/github/WeightFormer/EIMSdata/weights_train.pickle','rb') as f:
        weights_train = pickle.load(f)
    with open('E:/github/WeightFormer/EIMSdata/mz_list_test.pickle','rb') as f:
        mz_list_test = pickle.load(f)
    with open('E:/github/WeightFormer/EIMSdata/intensity_list_test.pickle','rb') as f:
        intensity_list_test = pickle.load(f)
    with open('E:/github/WeightFormer/EIMSdata/weights_test.pickle','rb') as f:
        weights_test = pickle.load(f)
    study_name = 'WeightFormer'
    study = optuna.create_study(study_name=study_name,direction="maximize")
    study.optimize(objective, n_trials=20)
    params = study.best_params