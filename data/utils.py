# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:16:57 2023

@author: Administrator
"""
import os
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
from model.Model import MyDataSet
from data.Data import DataTran,ProcessIndependent,LengthFilter,GetWeight,Pad_data
import torch
from model.TrainModel import Predict

def PlotResults(true_weights,predict_weights,title):
    '''
    Draw scatter plots of predicted and true results
    '''
    plt.figure()
    plt.scatter(true_weights,predict_weights,s=0.5)
    plt.xlabel('True Weights ')
    plt.ylabel('Predict Weights')
    diagonal = np.array([[0,max(true_weights)],[0,max(true_weights)]])
    plt.plot(diagonal[0,:],diagonal[1,:],color='r')
    plt.xlim(0,max((true_weights)))
    plt.ylim(0,max((true_weights)))
    plt.title(title)
    plt.show()

def CalWeights(true_weights,predict_weights):
    '''
    Calculate RMSE and MAE evaluation indicators
    '''
    rmse = np.sqrt(mean_squared_error(true_weights,predict_weights))
    mae = mean_absolute_error(true_weights,predict_weights)
    return rmse,mae

def Test2vec(mz_list_test,intensity_list_test):
    '''
    Convert the test set into the data types required for the PIM model
    '''
    nist_vec = []
    mz_list_test = [i.cpu().numpy() for i in mz_list_test]
    intensity_list_test = [i.cpu().numpy() for i in intensity_list_test]
    for i in tqdm(range(len(mz_list_test))):
        mz_ = mz_list_test[i]
        mz_ = mz_[mz_ > 0]
        intensity_ = intensity_list_test[i]
        intensity_ = intensity_[0:len(mz_)]
        nist_vec.append(np.vstack((mz_,intensity_)).T)
    return nist_vec

def BulidQSARData(mz_list_test,intensity_list_test):
    '''
    Convert the test set into the data types required for the QSAR model
    '''
    mz_list_test = [i.cpu().numpy() for i in mz_list_test]
    intensity_list_test = [i.cpu().numpy() for i in intensity_list_test]
    qsar_data = []
    for i in tqdm(range(len(mz_list_test))):
        mz_ = mz_list_test[i]
        mz_ = mz_[mz_ > 0]
        intensity_ = intensity_list_test[i]
        intensity_ = intensity_[0:len(mz_)]
        mz_ = np.append(-1,mz_)
        mz_ = pd.Series(mz_)
        intensity_ = np.append(sum(intensity_),intensity_)
        intensity_ = pd.Series(intensity_)
        info = pd.concat([mz_,intensity_],axis=1)
        info.columns = ['mz','intensity']
        qsar_data.append(info)
    qsar_data = pd.concat(qsar_data,axis=1)
    return qsar_data

def SaveQSARData(qsar_data):
    '''
    Store every 1000 spectra as cvs files
    '''
    start = 0
    end = start+2000
    while start < qsar_data.shape[1]:
        data = qsar_data.iloc[:,start:end]
        data.to_csv('EIMSdata/qsar/'+str(start)+'.csv',index=False)
        start += 2000
        end += 2000

def LoadQRSAPredData(qrsa_pred_file):
    '''
    Loading the test set predicted result by the QSAR model
    '''
    file_list = os.listdir(qrsa_pred_file)
    number = [float(i[0:-3]) for i in file_list]
    sort = np.argsort(number)
    mw_list = []
    file_list2 = [file_list[i] for i in sort]
    for file in file_list2:
        file_ = qrsa_pred_file+'/'+file
        mw_data = pd.read_csv(file_)
        mw = list(mw_data['Prediction'])
        mw_list = mw_list + mw
    return mw_list

def LoadPIMData(pim_data_file):
    '''
    Loading the test set predicted result by the PIM model
    '''
    pim_data = list(np.load(pim_data_file,allow_pickle=True))
    return pim_data

def CompareOther(weights_test,pred_result,title):
    '''
    Draw scatter plots of true data and predicted results obtained from other algorithms
    '''
    weights_test = [i.cpu().numpy() for i in weights_test]
    weights_test = [i.ravel()[0] for i in  weights_test]
    index = [i for i,v in enumerate(pred_result) if v != None]
    pred_result2 = [pred_result[i] for i in index]
    weights_test2 = [weights_test[i] for i in index]
    rmse = np.sqrt(mean_squared_error(weights_test2,pred_result2))
    mae = mean_absolute_error(weights_test2,pred_result2)
    PlotResults(weights_test2,pred_result2,title)
    return rmse,mae

def PredIndependent(model,independent_data,batch_size,maxlen):
    '''
    Using WeightFormer model on independent test set
    '''
    smiles_i,peak_vec_i = ProcessIndependent(independent_data)
    smiles_i,peak_vec_i = LengthFilter(smiles_i,peak_vec_i,maxlen)
    weights_i = GetWeight(smiles_i)
    mz_list_i,intensity_list_i = Pad_data(peak_vec_i,maxlen)
    mz_list_i = [torch.LongTensor(i) for i in mz_list_i]
    intensity_list_i = [torch.tensor(i) for i in intensity_list_i]
    weights_i = [torch.tensor(i) for i in weights_i]
    true_weights_i,predict_weights_i = Predict(model,mz_list_i,intensity_list_i,weights_i,batch_size)
    return smiles_i,true_weights_i,predict_weights_i

