#%% Imports and path check
import numpy as np
import sys
import os
import h5py

assert os.path.basename(os.getcwd()) == "calibration"
if "..\\" not in sys.path:sys.path.append("..\\")

from calib.input_data import InputData
from calib.analytical_calib import AnalyticalCalibration
#%% Choose dataset
dataset = "2022_02_24"
dataset_filename =  r"data\\" + dataset + r"\\Inputs\dotsTest.hdf5"

print("==========")
print("Using:")
print("Dataset:    ...", dataset)
print("Dots:       ...", dataset_filename[-30:])
print("==========")

#%% initialise input data structure

calib = AnalyticalCalibration()
inputdata = InputData()

#Load all data
inputdata.load_inputs(dataset_filename)
points,artefact = inputdata.get_inputs_numpy()

for i in range(len(points)):
    for j in range(len(points[i])):

        points[i][j] = np.concatenate((points[i][j], np.ones((points[0][0].shape[0],1))),axis=1)
        artefact[i][j][:,2] = 1

#%% Obtain analytical calibration
num_cameras = len(points)
K = [[] for _ in range(num_cameras)]
t = [[] for _ in range(num_cameras)]
r = [[] for _ in range(num_cameras)]

for i in range(num_cameras):

    if i==1:
        inv_fy=True
    else:
        inv_fy=False
    
    #obtain camera guess 
    K[i], t[i], r[i] = calib.calibrate(artefact[i], points[i], inv_fy=inv_fy)

#%% obtain extrinsics guess
extrinsic_r,extrinsic_t = calib.estimate_extrinsics(r,t)

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


