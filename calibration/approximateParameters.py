import numpy as np
import sys
import os
import h5py

currentDir = os.getcwd()
cutoff = currentDir.find("ProjectyBoy2000")
assert cutoff != -1
home = currentDir[:cutoff+16]

if home not in sys.path:sys.path.append(home)
if home+"Calibration" not in sys.path:sys.path.append(home+"Calibration")

from clib.AnalyticalCalibration import AnalyticalCalibration

from common import loadData

from commonlib.h5py_functions import (
    num_of_keys,
    )

#%%
dataset = "2021_09_12"

datasetFilename =  home + r"Calibration\Data\\" + dataset + r"\\Inputs\dots.hdf5"
boardFilename =  home + r"Calibration\Data\\" + dataset + r"\\Inputs\board.hdf5"

print("==========")
print("Using:")
print("Dots: ...", datasetFilename[-20:])
print("Board: ...", boardFilename[-20:])
print("==========")

#%% load in data

num_of_positions = num_of_keys(datasetFilename, "camera/points")

cPoints = []
pPoints = []

with h5py.File(boardFilename, 'r') as f:
    
    board = f["board"][:,:]
    
f.close()

board = np.concatenate((board, np.ones_like(board[:,0:1])), axis=1)
board = board.astype(np.float64)

with h5py.File(datasetFilename, 'r') as f:
    
    ones = np.ones((184,1))
    
    for i in range(0,num_of_positions):
        
        string = "{:02d}".format(i)
        
        cPoints.append(
            np.concatenate((f["camera/points"][string][:,:].astype(np.float64), ones), axis=1))
        pPoints.append(
            np.concatenate((f["projector/points"][string][:,:].astype(np.float64), ones), axis=1))

f.close()
board = np.concatenate((board[:,:2],board[:,3:]), axis=1)

#%% Obtain approximation

analyticalCalib = AnalyticalCalibration()

#obtain camera guess 
iKc, itc, irc = analyticalCalib.estimate(board, cPoints)

#obtain projector guess 
iKp, itp, irp = analyticalCalib.estimate(board, pPoints, inv_y=True)

#obtain extrinsics guess
ir,it = analyticalCalib.estimateExtrinsics(itc, irc, itp, irp)

#%% Assemble

def assembleVec(K, r, t):
    
    #Assemble parameter vector
    X = np.empty((5 + 6*t.shape[1]))
    
    X[0] = K[0,0]
    X[1] = K[1,1]
    X[2] = K[0,1]
    X[3] = K[0,2]
    X[4] = K[1,2]
    
        
    for i in range(t.shape[1]):
        
        extrinsic = np.concatenate((r[:,i], t[:,i]), axis=0)
        
        X[i*6+5:(i+1)*6+5] = extrinsic
        
    return X

camParams = assembleVec(iKc, irc, itc)
projParams = assembleVec(iKp, irp, itp)
extParams = np.concatenate((ir,it))

#%% save approximation

filename = home + r"Calibration\Data\\" + dataset + r"\\Parameter outputs\approx_seed.hdf5"
with h5py.File(filename, 'w-') as f:

        f.create_group("camera")
        f.create_group("projector")
        f.create_group("extrinsic")
        
        f.create_dataset("/camera/array", data=camParams)
        f.create_dataset("/projector/array", data=projParams)
        f.create_dataset("/extrinsic/array", data=extParams)


