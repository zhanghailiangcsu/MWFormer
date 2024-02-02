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
from matchms.importing import load_from_msp


def ProSingleMSP(single_msp_file):
    spectrum = list(load_from_msp(single_msp_file))[0]
    mz = spectrum.mz
    intensity = spectrum.intensities
    mz = np.array([int(i) for i in mz])
    intensity = np.array([float(i) for i in intensity])
    intensity = intensity/max(intensity)
    return mz,intensity

def ProBatchMSP(batch_msp_file):
    spectrum = list(load_from_msp(batch_msp_file))
    df = []
    for sp in spectrum:
        mz = sp.mz
        intensity = sp.intensities
        mz = np.array([int(i) for i in mz])
        intensity = np.array([float(i) for i in intensity])
        intensity = intensity/max(intensity)
        info = np.vstack((mz,intensity)).T
        info_list = []
        for j in range(info.shape[0]):
            unit = info[j,:]
            unit1 = str(int(unit[0]))
            unit2 = str(float(unit[1]))
            unit = unit1+':'+unit2
            info_list.append(unit)
        if len(info_list) < 1000:
            info_list.extend(['0:0.0']*(1000-len(info_list)))
        info_list = pd.Series(info_list)
        info_list = pd.concat([info_list],axis=1)
        info_list.columns = ['Spectrum']
        df.append(info_list)
    df = pd.concat(df,axis=1)
    return df
    
def ProSingle(peak_list):
    unit = peak_list.split(';')
    mz = []
    intensity = []
    for u in unit:
        u2 = u.split()
        mz.append(int(u2[0]))
        intensity.append(float(u2[1]))
    mz = np.array([int(i) for i in mz])
    intensity = np.array([float(i) for i in intensity])
    intensity = intensity/max(intensity)
    return mz,intensity

# def ProSingle(mz,intensity):
#     mz = mz.split(',')
#     mz = np.array([int(i) for i in mz])
#     intensity = intensity.split(',')
#     intensity = np.array([float(i) for i in intensity])
#     intensity = intensity/max(intensity)
#     return mz,intensity

def PlotMS(mz,intensity):
    fig, ax = plt.subplots()
    plt.vlines(mz,0,intensity)
    plt.hlines(0,0,max(mz)+10)
    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    st.pyplot(fig)


def MWpredict(model_file,mz,intensity):
    model = MWFormer(1000,128, 6, 16, 0)
    model.load_state_dict(torch.load(model_file,map_location=torch.device('cpu')))
    peak_vec = np.vstack((mz,intensity)).T
    mz,intensity = Pad_data([peak_vec],1000)
    mz = [torch.LongTensor(i) for i in mz]
    intensity = [torch.tensor(i) for i in intensity]
    _,predict_weights = Predict(model,mz,intensity,[torch.zeros(1)],1)
    return predict_weights[0]

def BatchPred(model_file,df):
    model = MWFormer(1000,128, 6, 16, 0)
    model.load_state_dict(torch.load(model_file,map_location=torch.device('cpu')))
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

def GUI():
    col1, col2 = st.columns([1,2])
    with col1:
        st.write("")
    with col2:
        st.image("logo.png",width=130)
    st.write("MWFormer: Direct Prediction of Molecular Weights from Electron Ionization Mass Spectra for Difficult to identify Compounds")
    app_mode = st.sidebar.selectbox('Select mode',['Single mode','Batch mode'])
    
    if app_mode == 'Single mode':
        st.title('Single mode')
        st.subheader('Single Demo file')
        with open('Single.msp','rb') as my_file:
            st.download_button('Down single demo file(.msp)', data=my_file,
                               file_name = 'Single.msp')
        
        single_msp_file = st.file_uploader('Upload Single file(.msp)', type='msp',
                                           accept_multiple_files=False)
        # mz = st.text_area('m/z')
        # intensity = st.text_area('Intensity')
        if st.button('File Predict'):
            if single_msp_file is not None:
                mz,intensity = ProSingleMSP(single_msp_file.name)
            else:
                st.write("No file uploaded.")
            mw_result = MWpredict('model/model.pkl',mz,intensity)
            mw_result = np.round(mw_result,2)
            col1, col2 = st.columns([1,2])
            with col1:
                st.write('The molecular weight predicted by MWFormer is',mw_result)
            with col2:
                PlotMS(mz,intensity)
        example_peaklist = '273 22;289 107;290 14;291 999;292 162;293 34;579 37;580 15'
        peak_list = st.text_area('Peak List',value = example_peaklist)
        
        if st.button('Peak List Predict'):
            col1, col2 = st.columns([1,2])
            mz,intensity = ProSingle(peak_list)
            mw_result = MWpredict('model/model.pkl',mz,intensity)
            mw_result = np.round(mw_result,2)
            with col1:
                st.write('The molecular weight predicted by MWFormer is',mw_result)
            with col2:
                PlotMS(mz,intensity)
        
    elif app_mode == 'Batch mode':
        st.title('Batch mode')
        st.subheader('Batch demo file') 
        with open('Batch.msp','rb') as my_file:
            st.download_button('Down batch demo file(.msp)', data=my_file,
                               file_name = 'Batch.msp')
    
        st.subheader('Predict file')  
        uploaded_file = st.file_uploader('Upload file(.msp)', type='msp',
                                         accept_multiple_files=False)
        if st.button('Predict'):
            if uploaded_file is not None:
                df = ProBatchMSP(uploaded_file.name)
            else:
                st.write("No file uploaded.")
            result = BatchPred('model/model.pkl',df)
            result = pd.DataFrame(result)
            result.columns = ['Predict Weights']
            result = result.to_excel('result.xlsx',index=False)
            st.subheader('Result file')
            with open('result.xlsx','rb') as f:
                st.download_button('Down predict file(.xlsx)', data=f,
                                   file_name = 'Result.xlsx')

if __name__ == '__main__':
    
    GUI()
    

    
    
