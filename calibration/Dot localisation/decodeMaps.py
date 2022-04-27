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

#%%

dataset = "2021_09_12"

datasetDir = home + r"\Calibration\Data\\" + dataset + "\\Raw Data\\"
datasetList = listdir(datasetDir, lambda x: x[-5:]=='.hdf5')
        
outputFilename = home + r"\Calibration\Data\\" + dataset + "\\Inputs\\phaseMaps.hdf5"
        
#%%
decode = Decode(True)
freq = [1/19, 1/21, 1/23]

#%%
"""
decodeType = filtered - filtering 0.4
decoedType = common - no filtering
"""
decodeType = "filtered"

print("decoding - ", decodeType)
#%%
for i in range(len(datasetList)):

    datasetString = datasetDir + datasetList[i]

    blankImage = load_h5py_arrays(datasetString, 0)
    
    print('=== {:02d} ==='.format(i))
    
    print('Decoding...')
    if decodeType == 'filtered':
        mappingX = decode.modifiedHeterodyne3stepFilter(datasetString,freq,offsetIndex=1,r=0.4).get()
        mappingY = decode.modifiedHeterodyne3stepFilter(datasetString,freq,offsetIndex=10,r=0.4).get()
    elif decodeType == 'common':
        mappingX = decode.modifiedHeterodyne3step(datasetString,freq,offsetIndex=1).get()
        mappingY = decode.modifiedHeterodyne3step(datasetString,freq,offsetIndex=10).get()
    else: raise Exception
    
    with h5py.File(outputFilename, 'a') as f:
        
        string = "{:02d}".format(i)
        
        if decodeType+"//"+string in f.keys():
            
            try:
                del f[decodeType+"//"+string]
            except: 
                pass
            
        f.create_dataset(decodeType+"//"+string+"//X", data=mappingX, compression='lzf')
        f.create_dataset(decodeType+"//"+string+"//Y", data=mappingY, compression='lzf')  
        
