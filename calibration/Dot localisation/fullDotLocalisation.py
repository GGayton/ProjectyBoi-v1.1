#%%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

if r"..\\..\\" not in sys.path: sys.path.append(r"..\\..\\")

from commonlib.file_finder import list_dir
from commonlib.h5py_functions import trydel

from DotLocalisation import DotLocalisation
#%% Choose datatsets

dataset = "2022_02_24"
datadir = r"..\\data\\"

phasemap_dataset = datadir + dataset + r"\\Inputs\\phaseMaps.hdf5"
decodeType = "filtered"

raw_dataset = datadir + dataset + "\\Raw Data\\"
raw_list = list_dir(raw_dataset, lambda x: x[-5:]=='.hdf5')

output_filename = datadir + dataset + r"\\Inputs\\dotsTest.hdf5"

board_filename = datadir + dataset + r"\\Inputs\\board.hdf5"
        
#%% Initialise
localiser = DotLocalisation()

with h5py.File(board_filename, 'r') as f:
    board = f["board"][()]

#%% localise all points
for i in range(3):#len(raw_list)):
    
    print('=== {:02d} ==='.format(i))
    
    print('Load blank image...')
    dataset_string = raw_dataset + raw_list[i]
    with h5py.File(dataset_string, 'r') as f:

        blank_image = f["00"][()]
        
    print('Load phase maps...')
    with h5py.File(phasemap_dataset, 'r') as f:
        mappingX = f[decodeType+"//{:02d}//X".format(i)][()]
        mappingY = f[decodeType+"//{:02d}//Y".format(i)][()]
       
    print('Localising camera points...')
    cParams,V,sigma,sigmaStd = localiser.localise(blank_image)
    
    print('Inferring projector points...')
    pParams,W = localiser.infer(cParams,V,mappingX,mappingY)
    
    with h5py.File(output_filename, 'a') as f:
        
        string = "{:02d}".format(i)
        
        if string+r"\\camera\\points" in f.keys():
            
            trydel(f,string+r"\\camera\\points")
            trydel(f,string+r"\\camera\\A")
            trydel(f,string+r"\\camera\\B")
            trydel(f,string+r"\\camera\\theta")
            trydel(f,string+r"\\camera\\sigma")
            trydel(f,string+r"\\camera\\sigmaStd")
            trydel(f,string+r"\\camera\\covariance")
            trydel(f,string+r"\\projector\\points")
            trydel(f,string+r"\\projector\\covariance")
            trydel(f,string+r"\\board\\points")
            
        f.create_dataset(string+r"\\camera\\points", data=cParams[:,:2], compression='lzf')
        f.create_dataset(string+r"\\camera\\A", data=cParams[:,2])  
        f.create_dataset(string+r"\\camera\\B", data=cParams[:,3])  
        f.create_dataset(string+r"\\camera\\theta", data=cParams[:,4])
        f.create_dataset(string+r"\\camera\\sigma", data=sigma)  
        f.create_dataset(string+r"\\camera\\sigmaStd", data=sigmaStd)
        f.create_dataset(string+r"\\camera\\covariance", data=V, compression='lzf')

        f.create_dataset(string+r"\\projector\\points", data=pParams[:,:2], compression='lzf')  
        f.create_dataset(string+r"\\projector\\covariance", data=W, compression='lzf')

        f.create_dataset(string+r"\\board\\points", data=board, compression='lzf')

    
    print('==========')

# %%
