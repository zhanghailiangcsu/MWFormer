# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 12:38:34 2023

@author: Administrator
"""
import os
os.chdir('E:/github/WeightFormer')
import numpy as np
from tqdm import tqdm
from rdkit import Chem  
from rdkit.Chem import Descriptors

def Peak2Vec(peak):
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
    smiles = list(ei_ms['smiles'])
    peak_vec = []
    for i in tqdm(range(len(ei_ms))):
        peak = ei_ms.iloc[i,:][1]
        vec = Peak2Vec(peak)
        peak_vec.append(vec)
    return smiles,peak_vec

def GetWeight(smiles):
    weights = []
    for s in tqdm(smiles):
        mol = Chem.MolFromSmiles(s)
        mol_weight = Descriptors.MolWt(mol) 
        weights.append(mol_weight)
    return weights
    
def LengthFilter(smiles,peak_vec):
    index = [i for i,v in enumerate(peak_vec) if len(v) < 200]
    smiles_new = [smiles[i] for i in index]
    peak_vec_new = [peak_vec[i] for i in index]
    return smiles_new,peak_vec_new

def Pad_data(peak_vec,max_len=200):
    mz_list = []
    intensity_list = []
    p = peak_vec[0]
    for p in tqdm(peak_vec):
        length = p.shape[0]
        if length < max_len:
            pad = max_len-length
            mz = np.append(p[:,0],np.zeros(pad))
            intensity = np.append(p[:,1],np.zeros(pad))
            mz_list.append(mz)
            intensity_list.append(intensity)
    return mz_list,intensity_list




