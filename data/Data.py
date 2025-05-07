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
def Peak2Vecnoise(peak):
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
    #I/= max(I)
            #delete noise
    keep = np.where(intensity > 0.01)[0]
    mz = mz[keep]
    intensity = intensity[keep]
    vec = np.vstack((mz,intensity)).T
    return vec
# smiles,peak_vec = DataTran(ei_ms)
def DataTran(ei_ms):
    '''
    Load NIST 2017 dataset
    '''
    smiles = list(ei_ms['SMILES'])
    peak_vec = []
    for i in tqdm(range(len(ei_ms))):
        peak = ei_ms.iloc[i,:][1]
        vec = Peak2Vec(peak)
        peak_vec.append(vec)
    return smiles,peak_vec

def GetWeight(smiles):
    '''
    Get molecular weight from SMILES
from rdkit import Chem  
from rdkit.Chem import Descriptors
s='CC(C)COC(=O)CCCC(=O)OCc1ccccc1C(F)(F)F'
mol = Chem.MolFromSmiles(s)
mol1= Chem.AddHs(mol)
print("mol Smiles:",Chem.MolToSmiles(mol))
print("mol1 Smiles:",Chem.MolToSmiles(mol1))
print("num ATOMs:",mol.GetNumAtoms())
print("num ATOMs:",mol1.GetNumAtoms())
mol_weight = Descriptors.ExactMolWt(mol1) 
print(mol_weight)
mol_weight2 = Descriptors.ExactMolWtt(mol1) 
print(mol_weight2)
    '''
    weights = []
    for s in tqdm(smiles):
        mol = Chem.MolFromSmiles(s)
        #mol1= Chem.AddHs(mol)
        mol_weight = Descriptors.ExactMolWt(mol) 
        #mol_weight2 = Descriptors.MolWt(mol1) 
        weights.append(mol_weight)
    return weights
def GetWeight1(ei_ms):
    '''
    Get molecular weight from SMILES
    '''
    weights = list(ei_ms['weight'])
   
    return weights
    
def LengthFilter(smiles,peak_vec,maxlen):
    '''
    Filter data using length filters
    '''
    index = [i for i,v in enumerate(peak_vec) if len(v) < maxlen]
    smiles_new = [smiles[i] for i in index]
    peak_vec_new = [peak_vec[i] for i in index]
    return smiles_new,peak_vec_new

def Pad_data(peak_vec,maxlen):
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


