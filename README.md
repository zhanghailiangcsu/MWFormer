# MWFormer  <div align=center><img src="https://github.com/zhanghailiangcsu/WeightFormer/blob/main/logo.png" width="202" height="213"/></div>  
MWFormer: Direct Prediction of Molecular Mass from Electron lonization Mass Spectrum by Transformer
# 1. Introduction
# 2. Depends
1.[Anaconda](https://www.anaconda.com) for Python 3.9   
2.[Pytorch](https://pytorch.org/) 1.12   
# 3. Install
1.Install anaconda   
2.Install [Git](https://git-scm.com/downloads)  
3.Open commond line, create environment and enter with the following commands.   
```
conda create -n MWFormer python=3.9  
conda activate MWFormer  
```
4.Clone the repository and enter.  
```
git clone https://github.com/zhanghailiangcsu/MWFormer.git
```
5.Install dependency with the following commands.
```
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install -c conda-forge rdkit
```
# 4. Usage
The MWFormer is public at [homepage](https://github.com/zhanghailiangcsu), every user can download and use it.
You can download the trained model on Github.
Then refer to the example to use the model for prediction, and directly obtain the molecular weight from EI-MS.
# 5. Contact
Hailiang Zhang   
E-mail 2352434994@qq.com
