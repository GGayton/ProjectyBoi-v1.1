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

DATATYPE = tf.float64

currentDir = os.getcwd()
cutoff = currentDir.find("ProjectyBoy2000")
assert cutoff != -1
home = currentDir[:cutoff+16]

if home not in sys.path:sys.path.append(home)
if home+"Calibration" not in sys.path:sys.path.append(home+"Calibration")

from ModelF import NonLinearCalibration,InputData
from seperationIndex import seperationIndex
from common import commonInputs, weightedRegressionCov,robustRegressionCov,regressionCov

#%%
dataset = "2022_02_24"
datasetFilename =  home + r"Calibration\Data\\" + dataset + r"\\Inputs\dotsCorrected.hdf5"
boardFilename =  home + r"Calibration\Data\\" + dataset + r"\\Inputs\board.hdf5"
estimateFilename = home + r"Calibration\Data\\" + dataset + r"\\Parameter outputs\\modelA_seed.hdf5"

print("==========")
print("Using:")
print("Dataset:    ...", dataset)
print("Dots:       ...", datasetFilename[-30:])
print("Board:      ...", boardFilename[-30:])
print("Estimate:   ...", estimateFilename[-30:])
print("==========")

#%% initialise input data structure

inputData = InputData()

I = seperationIndex()

#Load all data
inputData.loadBoardPoints(boardFilename, I)
inputData.loadMeasuredPoints(datasetFilename, I)
inputData.loadEstimate(estimateFilename, 0)

cBoardInput, pBoardInput, camInput, projInput = inputData.getInput1D()
initCamParams = inputData.getInitCamParams()
initProjParams = inputData.getInitProjParams()
initExtParams = inputData.getInitExtParams()

cBoardInput = cBoardInput.numpy()
pBoardInput = pBoardInput.numpy()

plt.figure()
plt.plot(cBoardInput[0,:], cBoardInput[1,:], '.')
plt.plot(pBoardInput[0,:], pBoardInput[1,:], 'x')

cBoardInput[2,:] = 0
pBoardInput[2,:] = 0

cBoardInput = tf.constant(cBoardInput,dtype = DATATYPE)
pBoardInput = tf.constant(pBoardInput,dtype = DATATYPE)

#%% Initialise

cNum = cBoardInput.shape[1]
pNum = pBoardInput.shape[1]

positionsNum = (camInput.shape[0]//2)//cNum

camCalib = NonLinearCalibration(cNum, positionsNum)
projCalib = NonLinearCalibration(pNum, positionsNum)

#%% clib

dampingFactor, DISPLAY, ITERATION_MAX, MIN_CHANGE, FAILURE_COUNT_MAX = commonInputs(DATATYPE)

L = 0.95
H = 1.05

optimCamParams, Jc, residualsC = camCalib.optimise(
    cBoardInput,
    camInput,
    initCamParams,# * tf.random.uniform(initCamParams.shape, minval = L, maxval = H, dtype = DATATYPE),
    dampingFactor,
    DISPLAY,
    FAILURE_COUNT_MAX,
    MIN_CHANGE,
    ITERATION_MAX)
print(np.sum(residualsC.numpy()**2)**0.5)
optimProjParams, Jp, residualsP = projCalib.optimise(
    pBoardInput,
    projInput,
    initProjParams * tf.random.uniform(initProjParams.shape, minval = L, maxval = H, dtype = DATATYPE),
    dampingFactor,
    DISPLAY,
    FAILURE_COUNT_MAX,
    MIN_CHANGE,
    ITERATION_MAX)
print(np.sum(residualsP.numpy()**2)**0.5)
[126]

#%%



[11.35]
[2.64]


#%%

print("Deriving final covariance matrix... ", end="")

sc,_ = robustRegressionCov(Jc.numpy().astype(float),residualsC.numpy().astype(float))
sp,_ = robustRegressionCov(Jp.numpy().astype(float),residualsP.numpy().astype(float))

print("done")
#%%
from clib.ExtrinsicEstimate import estimate3, testSelfConsistency3

nI = camCalib.nI

r,t,V,rFull,tFull,Vrt = estimate3(optimCamParams[nI:], optimProjParams[nI:], sc[nI:, nI:], sp[nI:, nI:])

optimExtParams = np.concatenate((r,t),axis=0)
print("Self-consistency test")
testSelfConsistency3(r,t,V,rFull,tFull,Vrt)

#%%
from clib.ExtrinsicEstimate import printResults
import matplotlib.pyplot as plt
plt.close('all')
printResults(rFull,tFull,Vrt)

#%%
nI = camCalib.nI
cov = np.zeros((nI*2+6,nI*2+6))
cov[:nI,:nI] = sc[:nI,:nI]
cov[nI:nI*2,nI:nI*2] = sp[:nI,:nI]
cov[nI*2:,nI*2:] = V


#%% Save data

if True:
    camParams = optimCamParams.numpy()
    projParams = optimProjParams.numpy()
    extParams = optimExtParams

    
    filename = home +"Calibration\\Data\\" + dataset + r"\\Parameter outputs\\modelF_seed.hdf5"
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
                
        f.create_dataset("/camera/jacobian", data=Jc.numpy(), compression = 'lzf')
        f.create_dataset("/projector/jacobian", data=Jp.numpy(), compression = 'lzf')
        
        f.create_dataset("/camera/residuals", data=residualsC.numpy(), compression = 'lzf')
        f.create_dataset("/projector/residuals", data=residualsP.numpy(), compression = 'lzf')
        
        f.create_dataset("covariance", data=cov, compression = 'lzf')
        f.create_dataset("/camera/full covariance", data=sc, compression = 'lzf')
        f.create_dataset("/projector/full covariance", data=sp, compression = 'lzf')
        f.create_dataset("/extrinsic/full covariance", data=V, compression = 'lzf')



