
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py

currentDir = os.getcwd()
cutoff = currentDir.find("ProjectyBoy2000")
assert cutoff!=-1
home = currentDir[:cutoff+16]

if home not in sys.path: sys.path.append(home)
if home+"Calibration" not in sys.path: sys.path.append(home+"Calibration")
if home+"Calibration\Dot localisation v2" not in sys.path: sys.path.append(home+"Calibration\Dot localisation v2")
if home+"Phase decoding" not in sys.path:sys.path.append(home+"Phase decoding")

from DotLocalisation import DotLocalisation

from Decoding import Decode

from commonlib.h5py_functions import load_h5py_arrays, tryDel

from commonlib.common_functions import (
    listdir,
    plotErrorEllipse
    )

    
#%% Choose datasets

dataset = "2022_02_24"

datasetDir = home + r"\Calibration\Data\\" + dataset + "\\Raw Data\\"
datasetList = listdir(datasetDir, lambda x: x[-5:] == ".hdf5")

phaseMapDataset = home + r"\Calibration\Data\\" + dataset + "\\Inputs\\phaseMaps.hdf5"



dotFilename = home + r"\Calibration\Data\\" + dataset + "\\Inputs\\dots.hdf5"

#%% Load data

cParams = []
with h5py.File(dotFilename, 'r') as f:
    nPos = len(list(f["camera//points"].keys()))
    for i in range(0,nPos):
        
        string = "{:02d}".format(i)
        
        points = f["camera//points//{:02d}".format(i)][()].reshape(-1,2)
        A = f["camera//A//{:02d}".format(i)][()].reshape(-1,1)
        B = f["camera//B//{:02d}".format(i)][()].reshape(-1,1)
        T = f["camera//theta//{:02d}".format(i)][()].reshape(-1,1)
        
        cParams.append(np.concatenate((points, A, B, T),axis=1))
        
        
#%% Initialise
dotLocaliser = DotLocalisation()

#%%
plt.close('all')
k = np.random.randint(0,25)
j = np.random.randint(184)

k=20
j=88

decodeType = 'filtered'
print("Decoding as ", decodeType)

with h5py.File(phaseMapDataset, 'r') as f:
        mappingX = f[decodeType+"//{:02d}//X".format(k)][()]
        mappingY = f[decodeType+"//{:02d}//Y".format(k)][()]    

datasetString = datasetDir + datasetList[k]    

blankImage = load_h5py_arrays(datasetString, 0)

print('=== {:02d}//{:02d} ==='.format(k,j))
dotLocaliser.troubleshootLocalisation(blankImage,j)

dotLocaliser.troubleshootInference(cParams[k],mappingX, mappingY,j)


#%%

dot = np.random.randn(2).flatten()

V = np.random.rand(2,2)
V = V@V.T

plt.close('all')
plotErrorEllipse(dot,V)
