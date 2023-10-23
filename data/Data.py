# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 12:38:34 2023

@author: Administrator
"""
import numpy as np
from tqdm import tqdm
from rdkit import Chem  
from rdkit.Chem import Descriptors

def Peak2Vec(peak):
    '''
    Convert the peak of a string to vector
    '''
    peak = peak.split()
    mz = np.array([])
    intensity = np.array([])
    for p in peak:
        p = p.split(':')
        mz = np.append(mz, float(p[0]))
        intensity = np.append(intensity, float(p[1]))
    intensity = intensity/max(intensity)
    vec = np.vstack((mz,intensity)).T
    return vec

def DataTran(ei_ms):
    '''
    Load NIST 2017 dataset
    '''
    smiles = list(ei_ms['smiles'])
    peak_vec = []
    for i in tqdm(range(len(ei_ms))):
        peak = ei_ms.iloc[i,:][1]
        vec = Peak2Vec(peak)
        peak_vec.append(vec)
    return smiles,peak_vec

def GetWeight(smiles):
    '''
    Get molecular weight from SMILES
    '''
    weights = []
    for s in tqdm(smiles):
        mol = Chem.MolFromSmiles(s)
        mol_weight = Descriptors.MolWt(mol) 
        weights.append(mol_weight)
    return weights
    
def LengthFilter(smiles,peak_vec,maxlen):
    '''
    Filter data using length filters
    '''
    index = [i for i,v in enumerate(peak_vec) if len(v) < maxlen]
    smiles_new = [smiles[i] for i in index]
    peak_vec_new = [peak_vec[i] for i in index]
    return smiles_new,peak_vec_new

def Pad_data(peak_vec,maxlen=200):
    '''
    Pad data to same dimension
    '''
    mz_list = []
    intensity_list = []
    p = peak_vec[0]
    for p in tqdm(peak_vec):
        length = p.shape[0]
        if length < maxlen:
            pad = maxlen-length
            mz = np.append(p[:,0],np.zeros(pad))
            intensity = np.append(p[:,1],np.zeros(pad))
            mz_list.append(mz)
            intensity_list.append(intensity)
    return mz_list,intensity_list

def ProcessIndependent(independent_data):
    '''
    Load Independent dataset
    '''
    smiles =  list(independent_data['SMILES'])
    peak_vec = []
    spectrums = list(independent_data['Spectramain'])
    s = spectrums[0]
    for s in tqdm(spectrums):
        vec = Peak2Vec(s)
        peak_vec.append(vec)
    return smiles,peak_vec


