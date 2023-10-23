# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 15:42:51 2023

@author: Administrator
"""
import os
os.chdir('E:/github/WeightFormer')
from data.Data import DataTran,ProcessIndependent,LengthFilter,GetWeight
from model.TrainModel import Predict,PlotLoss,Train,dataset_sep
from data.Data import LengthFilter,GetWeight,Pad_data
import torch
import pickle
from model.Model import WeightFormer
from data.utils import PlotResults,CalWeights,Test2vec,BulidQSARData,SaveQSARData
from data.utils import LoadQRSAPredData,CompareOther,LoadPIMData,PredIndependent
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def GetTestSMILES(smiles,val_size = 0.2):
    np.random.seed(0)
    n = len(smiles)
    perm = np.random.permutation(n)
    n_train = int(n*(1-val_size))
    perm_val = perm[n_train:]
    smiles_test = [smiles[x] for x in perm_val]
    return smiles_test

def MergeData(smiles_test,weights_test,predict_weights,qsar_result,pim_result):
    weights_test = [i.cpu().numpy() for i in weights_test]
    weights_test = [i.ravel()[0] for i in weights_test]
    df = pd.DataFrame()
    df['SMILES'] = smiles_test
    df['True Weights'] = weights_test
    df['WeightFormer'] = predict_weights
    df['QSAR'] = qsar_result
    df['PIM'] = pim_result
    return df

def SaveIndepend(name):
    df = pd.DataFrame()
    df['SMILES'] = smiles_i
    df['True Weights'] = true_weights_i
    df['WeightFormer'] = predict_weights_i
    df.to_csv('EIMSdata/'+name,index=False)


if __name__ == '__main__':
    
    batch_size = 64
    lr = 5e-4
    epochs = 10
    vocab_size = 1000
    hidden = 128 
    n_layers = 6
    attn_heads = 8
    dropout = 0
    maxlen = 200
    
    #Load NIST2017 Dataset
    file  = 'E:/NISTEIMS/EI-MS-246809.csv'
    ei_ms = pd.read_csv(file)
    smiles,peak_vec = DataTran(ei_ms)
    smiles,peak_vec = LengthFilter(smiles,peak_vec,maxlen)
    weights = GetWeight(smiles)
    mz_list,intensity_list = Pad_data(peak_vec,maxlen)
    mz_list = [torch.LongTensor(i) for i in mz_list]
    intensity_list = [torch.tensor(i) for i in intensity_list]
    weights = [torch.tensor(i) for i in weights]
    
    # Randomly divide the dataset into training and testing sets and save them
    mz_list_train,intensity_list_train,weights_train,mz_list_test,intensity_list_test,weights_test= dataset_sep(mz_list,intensity_list,weights,0.2)
    pickle.dump(mz_list_train, open('EIMSdata/mz_list_train.pickle','wb'))
    pickle.dump(intensity_list_train, open('EIMSdata/intensity_list_train.pickle','wb'))
    pickle.dump(weights_train, open('EIMSdata/weights_train.pickle','wb'))
    pickle.dump(mz_list_test, open('EIMSdata/mz_list_test.pickle','wb'))
    pickle.dump(intensity_list_test, open('EIMSdata/intensity_list_test.pickle','wb'))
    pickle.dump(weights_test, open('EIMSdata/weights_test.pickle','wb'))
    
    # Load saved data
    # with open('EIMSdata/mz_list_train.pickle','rb') as f:
    #     mz_list_train = pickle.load(f)
    # with open('EIMSdata/intensity_list_train.pickle','rb') as f:
    #     intensity_list_train = pickle.load(f)
    # with open('EIMSdata/weights_train.pickle','rb') as f:
    #     weights_train = pickle.load(f)
    # with open('EIMSdata/mz_list_test.pickle','rb') as f:
    #     mz_list_test = pickle.load(f)
    # with open('EIMSdata/intensity_list_test.pickle','rb') as f:
    #     intensity_list_test = pickle.load(f)
    # with open('EIMSdata/weights_test.pickle','rb') as f:
    #     weights_test = pickle.load(f)
    
    #Laod trained WeightFormer model
    model_file = 'model/model.pkl'
    model = WeightFormer(vocab_size,hidden, n_layers, attn_heads, dropout)
    model.load_state_dict(torch.load(model_file))
    
    # Using WeightFormer to predict the molecular weight of the test set and display it
    true_weights,predict_weights = Predict(model,mz_list_test,intensity_list_test,weights_test,batch_size)
    PlotResults(true_weights,predict_weights,'WeightFormer')
    rmse,mae = CalWeights(true_weights,predict_weights)
    
    # Building data compared to other methods
    nist_vec = Test2vec(mz_list_test,intensity_list_test)
    pickle.dump(nist_vec, open('EIMSdata/test_data.pickle','wb'))
    qsar_data = BulidQSARData(mz_list_test,intensity_list_test)
    SaveQSARData(qsar_data)
    
    # Load prediction results from other methods
    qrsa_pred_file = 'EIMSdata/result'
    qsar_result = LoadQRSAPredData(qrsa_pred_file)
    rmse_q,mae_q = CompareOther(weights_test,qsar_result,'QSAR')
    
    pim_data_file = 'EIMSdata/pred_mass.npy'
    pim_result = LoadPIMData(pim_data_file)
    rmse_p,mae_p = CompareOther(weights_test,pim_result,'PIM')
    
    # Merge results on test set
    smiles_test = GetTestSMILES(smiles,val_size = 0.2)
    df = MergeData(smiles_test,weights_test,predict_weights,qsar_result,pim_result)
    df.to_csv('EIMSdata/merge.csv',index=False)
    
    # Testing WeightFormer on independent test set
    independent_file = 'EIMSdata/test_11499.csv'
    independent_data = pd.read_csv(independent_file)
    smiles_i,true_weights_i,predict_weights_i = PredIndependent(model,independent_data,batch_size)
    rmse_i,mae_i = CalWeights(true_weights_i,predict_weights_i)
    PlotResults(true_weights_i,predict_weights_i,'Independent data')
    SaveIndepend('independent_result.csv')
    


    
