# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 09:43:56 2023

@author: Administrator
"""
# streamlit run stream.py
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
from model.Model import MWFormer
import torch
from model.TrainModel import Predict
from data.Data import Pad_data
import pandas as pd

def ProSingle(mz,intensity):
    mz = mz.split(',')
    mz = np.array([int(i) for i in mz])
    intensity = intensity.split(',')
    intensity = np.array([float(i) for i in intensity])
    intensity = intensity/max(intensity)
    return mz,intensity

def PlotMS(mz,intensity):
    fig, ax = plt.subplots()
    plt.vlines(mz,0,intensity)
    plt.hlines(0,0,max(mz)+10)
    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    st.pyplot(fig)


def MWpredict(model_file,mz,intensity):
    model = MWFormer(1000,256, 6, 8, 0)
    model.load_state_dict(torch.load(model_file))
    peak_vec = np.vstack((mz,intensity)).T
    mz,intensity = Pad_data([peak_vec],1000)
    mz = [torch.LongTensor(i) for i in mz]
    intensity = [torch.tensor(i) for i in intensity]
    _,predict_weights = Predict(model,mz,intensity,[torch.zeros(1)],1)
    return predict_weights[0]

def BatchPred(model_file,df):
    model = MWFormer(1000,256, 6, 8, 0)
    model.load_state_dict(torch.load(model_file))
    mz_list = []
    intensity_list = []
    for i in range(df.shape[1]):
        info = list(df.iloc[:,i])
        info = [i.split(':') for i in info]
        mz = np.array([int(i[0]) for i in info])
        intensity = np.array([float(i[1]) for i in info])
        mz_list.append(torch.LongTensor(mz))
        intensity_list.append(torch.tensor(intensity))
    if len(mz_list) <= 32:
        batch = len(mz_list)
    else:
        batch = 32
    true_w = [torch.zeros(1)[0] for i in range(len(mz_list))]
    _,predict_weights = Predict(model,mz_list,intensity_list,true_w,batch)
    return predict_weights

col1, col2 = st.columns([1,2])
with col1:
    st.write("")
with col2:
    st.image("logo.png",width=130)
st.write("MWFormer: Direct Prediction of Molecular Weights from Electron Ionization Mass Spectra for Difficult to identify Compounds")
app_mode = st.sidebar.selectbox('Select mode',['Single mode','Batch mode'])

if app_mode == 'Single mode':
    st.title('Single mode')
    mz = st.text_area('m/z')
    intensity = st.text_area('Intensity')
    if st.button('Predict'):
        mz,intensity = ProSingle(mz,intensity)
        mw_result = MWpredict('model/model.pkl',mz,intensity)
        
        col1, col2 = st.columns([1,2])
        with col1:
            st.write('The molecular weight predicted by MWFormer is',mw_result)
        with col2:
            PlotMS(mz,intensity)
    
elif app_mode == 'Batch mode':
    st.title('Batch mode')
    st.subheader('Demo file') 
    with open('demo.xlsx','rb') as my_file:
        st.download_button('Down demo file(.xlsx)', data=my_file,
                           file_name = 'Demo.xlsx')

    st.subheader('Predict file')  
    uploaded_file = st.file_uploader('Upload file(.xlsx)', type='xlsx')
    if st.button('Predict'):
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
        else:
            st.write("No file uploaded.")
        result = BatchPred('model/model.pkl',df)
        result = pd.DataFrame(result)
        result.columns = ['Predict Weights']
        result = result.to_excel('result.xlsx',index=False,encoding='utf-8-sig')
        st.subheader('Result file')
        with open('result.xlsx','rb') as f:
            st.download_button('Down predict file(.xlsx)', data=f,
                               file_name = 'Result.xlsx')
    
    
    
# df = []
# for i in range(1027,1030):
#     mz_i = mz_list_test[i].cpu().numpy()
#     intensity_i = intensity_list_test[i].cpu().numpy()
#     info = np.vstack((mz_i,intensity_i)).T
#     info_list = []
#     for j in range(info.shape[0]):
#         unit = info[j,:]
#         unit1 = str(int(unit[0]))
#         unit2 = str(float(unit[1]))
#         unit = unit1+':'+unit2
#         info_list.append(unit)
#     info_list = pd.Series(info_list)
#     info_list = pd.concat([info_list],axis=1)
#     info_list.columns = ['Spectrum']
#     df.append(info_list)
# df = pd.concat(df,axis=1)
# df.to_excel('demo.xlsx',index=False,encoding='utf-8-sig')
    
# p = peak_vec[0:10]
    
    
    
    
    
    
