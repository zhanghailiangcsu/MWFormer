# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 15:42:51 2023

@author: Administrator
"""
from data import DataTran
from TrainModel import Predict,PlotLoss,Train,dataset_sep
from data import LengthFilter,GetWeight,Pad_data
import torch
import pickle
from Model import WeightFormer
from utils import PlotResults,CalWeights,Test2vec,BulidQSARData,SaveQSARData
from utils import LoadQRSAPredData,CompareOther,LoadPIMData
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    batch_size = 64
    lr = 5e-4
    epochs = 10
    vocab_size = 1000
    hidden = 128 
    n_layers = 6
    attn_heads = 8
    dropout = 0
    
    
    model_file = 'model.pkl'
    model = WeightFormer(vocab_size,hidden, n_layers, attn_heads, dropout)
    model.load_state_dict(torch.load(model_file))
    # file  = 'E:/NISTEIMS/EI-MS-246809.csv'
    # ei_ms = pd.read_csv(file)
    # smiles,peak_vec = DataTran(ei_ms)
    # smiles,peak_vec = LengthFilter(smiles,peak_vec)
    # weights = GetWeight(smiles)
    # mz_list,intensity_list = Pad_data(peak_vec,max_len=200)
    # mz_list = [torch.LongTensor(i) for i in mz_list]
    # intensity_list = [torch.tensor(i) for i in intensity_list]
    # weights = [torch.tensor(i) for i in weights]
    
    # mz_list_train,intensity_list_train,weights_train,mz_list_test,intensity_list_test,weights_test= dataset_sep(mz_list,intensity_list,weights,0.2)
    # pickle.dump(mz_list_train, open('EIMSdata/mz_list_train.pickle','wb'))
    # pickle.dump(intensity_list_train, open('EIMSdata/intensity_list_train.pickle','wb'))
    # pickle.dump(weights_train, open('EIMSdata/weights_train.pickle','wb'))
    # pickle.dump(mz_list_test, open('EIMSdata/mz_list_test.pickle','wb'))
    # pickle.dump(intensity_list_test, open('EIMSdata/intensity_list_test.pickle','wb'))
    # pickle.dump(weights_test, open('EIMSdata/weights_test.pickle','wb'))
    
    with open('EIMSdata/mz_list_train.pickle','rb') as f:
        mz_list_train = pickle.load(f)
    with open('EIMSdata/intensity_list_train.pickle','rb') as f:
        intensity_list_train = pickle.load(f)
    with open('EIMSdata/weights_train.pickle','rb') as f:
        weights_train = pickle.load(f)
    with open('EIMSdata/mz_list_test.pickle','rb') as f:
        mz_list_test = pickle.load(f)
    with open('EIMSdata/intensity_list_test.pickle','rb') as f:
        intensity_list_test = pickle.load(f)
    with open('EIMSdata/weights_test.pickle','rb') as f:
        weights_test = pickle.load(f)
    
    
    true_weights,predict_weights = Predict(model,mz_list_test,intensity_list_test,weights_test,batch_size)
    PlotResults(true_weights,predict_weights)
    
    true_weights_train,predict_weights_train = Predict(model,mz_list_train,intensity_list_train,weights_train,batch_size)
    PlotResults(true_weights,predict_weights)
    
    
    rmse,mae = CalWeights(true_weights,predict_weights)
    
    nist_vec = Test2vec(mz_list_test,intensity_list_test)
    pickle.dump(nist_vec, open('EIMSdata/test_data.pickle','wb'))
    
    qsar_data = BulidQSARData(mz_list_test,intensity_list_test)
    SaveQSARData(qsar_data)
    
    qrsa_pred_file = 'EIMSdata/result'
    pred_result = LoadQRSAPredData(qrsa_pred_file)
    rmse_q,mae_q = CompareOther(weights_test,pred_result)
    
    
    pim_data_file = 'EIMSdata/pred_mass.npy'
    pim_data = LoadPIMData(pim_data_file)
    rmse_p,mae_p = CompareOther(weights_test,pim_data)
    


    
