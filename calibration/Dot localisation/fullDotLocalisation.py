import os
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np


currentDir = os.getcwd()
cutoff = currentDir.find("ProjectyBoy2000")
assert cutoff != -1
home = currentDir[:cutoff+16]

if home not in sys.path:sys.path.append(home)
if home+"Phase decoding" not in sys.path:sys.path.append(home+"Phase decoding")

from Decoding import Decode

from commonlib.h5py_functions import load_h5py_arrays
from commonlib.common_functions import listdir

from DotLocalisation import DotLocalisation
#%% Choose datatsets

dataset = "2021_09_12"

phaseMapDataset = home + r"\Calibration\Data\\" + dataset + "\\Inputs\\phaseMaps.hdf5"
decodeType = "filtered"

rawDataset = home + r"\Calibration\Data\\" + dataset + "\\Raw Data\\"
rawDatasetList = listdir(rawDataset, lambda x: x[-5:]=='.hdf5')

outputFilename = home + r"\Calibration\Data\\" + dataset + "\\Inputs\\dots.hdf5"
        
#%% Initialise
dotLocaliser = DotLocalisation()

#%%
for i in range(len(rawDatasetList)):
    
    print('=== {:02d} ==='.format(i))
    
    print('Load blank image...')
    datasetString = rawDataset + rawDatasetList[i]
    blankImage = load_h5py_arrays(datasetString, 0)
    
    # dotLocaliser.extractCameraPoints(blankImage)
    
    print('Load phase maps...')
    with h5py.File(phaseMapDataset, 'r') as f:
        mappingX = f[decodeType+"//{:02d}//X".format(i)][()]
        mappingY = f[decodeType+"//{:02d}//Y".format(i)][()]
       
    print('Localising camera points...')
    cParams,V,sigma,sigmaStd = dotLocaliser.localise(blankImage)
    
    print('Inferring projector points...')
    pParams,W = dotLocaliser.infer(cParams,V,mappingX,mappingY)
    
    with h5py.File(outputFilename, 'a') as f:
        
        string = "{:02d}".format(i)
        
        if "/camera/points/"+string in f.keys():
            
            try:
                del f["/camera/points/"+string]
                del f["/camera/A/"+string]
                del f["/camera/B/"+string]
                del f["/camera/theta/"+string]
                del f["/camera/sigma/"+string]
                del f["/camera/sigmaStd/"+string]
                del f["/camera/covariance/"+string]
                del f["/projector/points/"+string]
                del f["/projector/covariance/"+string]
            except: 
                pass
            
        f.create_dataset("/camera/points/"+string, data=cParams[:,:2])
        f.create_dataset("/camera/A/"+string, data=cParams[:,2])  
        f.create_dataset("/camera/B/"+string, data=cParams[:,3])  
        f.create_dataset("/camera/theta/"+string, data=cParams[:,4])
        f.create_dataset("/camera/sigma/"+string, data=sigma)  
        f.create_dataset("/camera/sigmaStd/"+string, data=sigmaStd)
        f.create_dataset("/camera/covariance/"+string, data=V)

        
        f.create_dataset("/projector/points/"+string, data=pParams[:,:2])  
        # f.create_dataset("/projector/A/"+string, data=pParams[:,2])  
        # f.create_dataset("/projector/B/"+string, data=pParams[:,3])  
        # f.create_dataset("/projector/theta/"+string, data=pParams[:,4])
        f.create_dataset("/projector/covariance/"+string, data=W)
    
    print('==========')

