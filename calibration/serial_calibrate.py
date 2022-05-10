#%% imports
import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys

if "..\\" not in sys.path:sys.path.append("..\\")

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

# from ModelF import NonLinearCalibration,InputData
# from seperationIndex import seperationIndex
# from common import commonInputs, weightedRegressionCov,robustRegressionCov,regressionCov
from calib.input_data import InputData

#%%
dataset = "2022_02_24"
dataset_filename =  r"data\\" + dataset + r"\\Inputs\dotsTest.hdf5"
estimate_filename = r"data\\" + dataset + r"\\Parameter outputs\\modelA_seed.hdf5"

print("="*30)
print("Using:")
print("Dataset:    ...", dataset)
print("Dots:       ...", dataset_filename[-30:])
print("Estimate:   ...", estimate_filename[-30:])
print("="*30)

#%% initialise input data structure

inputdata = InputData()

#Load all data
inputdata.load_inputs(dataset_filename)
out = inputdata.get_inputs_TF()

#%% test cell
from calib.serial import SerialCalibration
import time

c = SerialCalibration()


#%%


#%%

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



