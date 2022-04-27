import tensorflow as tf
import numpy as np
import time
import sys
import os
import h5py
import matplotlib.pyplot as plt

options = {
    'layout_optimizer': True, 
    'constant_folding': True, 
    'remapping': True, 
    'dependency_optimization': True, 
    'loop_optimization': True,
    'function_optimization': True, 
    'debug_stripper': False, 
    'disable_model_pruning': False, 
    'disable_meta_optimizer': False
}
tf.config.optimizer.set_experimental_options(options)

currentDir = os.getcwd()
cutoff = currentDir.find("ProjectyBoy2000")
assert cutoff != -1
home = currentDir[:cutoff+16]

if home not in sys.path:sys.path.append(home)
if home+"Calibration" not in sys.path:sys.path.append(home+"Calibration")

from ModelG import NonLinearCalibration,InputData
from common import commonInputs, robustRegressionCov

from seperationIndex import seperationIndex

#%%
dataset = "2022_02_24"
datasetFilename =  home + r"Calibration\Data\\" + dataset + r"\\Inputs\dotsCorrected.hdf5"
boardFilename =  home + r"Calibration\Data\\" + dataset + r"\\Inputs\board.hdf5"
estimateFilename = home + r"Calibration\Data\\" + dataset + r"\\Parameter outputs\modelA_seed.hdf5"
inputCovFilename = home + "Calibration\Data\\" + dataset + r"\\Covariances\inputCovG.hdf5"

print("==========")
print("Using:")
print("Dataset:    ...", dataset)
print("Dots:       ...", datasetFilename[-30:])
print("Board:      ...", boardFilename[-30:])
print("Estimate:   ...", estimateFilename[-30:])
print("Covariance: ...", inputCovFilename[-30:])
print("==========")

#%% initialise input data structure

inputData = InputData()

I = seperationIndex()

#Load all data
inputData.loadBoardPoints(boardFilename,I)
inputData.loadMeasuredPoints(datasetFilename,I)
inputData.loadEstimateFromModelA(estimateFilename)

cBoardInput, pBoardInput, camInput, projInput = inputData.getInput1D()
initParams = inputData.getInitParams()
pointsInput = tf.concat((camInput,projInput), axis=0)

#%% Initialise

cNum = cBoardInput.shape[1]
pNum = pBoardInput.shape[1]

positionsNum = (camInput.shape[0]//2)//cNum

DATATYPE = tf.float64

primaryCalib = NonLinearCalibration(cNum, pNum, positionsNum)


#%% weights

Wy = np.eye((cNum+pNum)*positionsNum*2)
Wy = tf.constant(Wy, dtype = DATATYPE)


#%% clib

dampingFactor, DISPLAY, ITERATION_MAX, MIN_CHANGE, FAILURE_COUNT_MAX = commonInputs(DATATYPE)

L = 0.98
H = 1.02

optimParams, J, residuals = primaryCalib.weightedOptimise(
    cBoardInput,
    pBoardInput,
    pointsInput,
    initParams * tf.random.uniform(initParams.shape, minval = L, maxval = H, dtype = DATATYPE),
    Wy,
    dampingFactor,
    DISPLAY,
    FAILURE_COUNT_MAX,
    MIN_CHANGE,
    ITERATION_MAX)

print(np.sum(residuals**2)**0.5)

[11.97]

#%%
print("Deriving final covariance matrix... ", end="")

V,_ = robustRegressionCov(J.numpy(),residuals.numpy())

print("done")

#%%

nI = primaryCalib.nI
cov = V[:2*nI+6,:2*nI+6]

#%% Save data

if True:
    camParams = optimParams[:nI].numpy()
    projParams =  optimParams[nI:2*nI].numpy()
    extParams = optimParams[2*nI:2*nI+6].numpy()

    
    filename = home +"Calibration\\Data\\" + dataset + r"\\Parameter outputs\\modelG_seed.hdf5"
    print("==========")
    print("Saving in: ...", filename[-50:])
    print("==========")
            
    with h5py.File(filename, 'a') as f:
        
        f.create_group("camera")
        f.create_group("projector")
        f.create_group("extrinsic")
        
        f.create_dataset("/camera/array", data=camParams, compression = 'lzf')
        f.create_dataset("/projector/array", data=projParams, compression = 'lzf')
        f.create_dataset("/extrinsic/array", data=extParams, compression = 'lzf')
                
        f.create_dataset("/cameraprojector/jacobian", data=J.numpy(), compression = 'lzf')
        
        f.create_dataset("/cameraprojector/residuals", data=residuals.numpy(), compression = 'lzf')
        
        f.create_dataset("covariance", data=cov, compression = 'lzf')
        f.create_dataset("full covariance", data=V, compression = 'lzf')
        f.create_dataset("full parameters", data=optimParams, compression = 'lzf')

