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

#%% Reprojection errors

#%% save
names = inputdata.keys
filename = r"data\\" + dataset + r"\\parameter outputs\approx.hdf5"
with h5py.File(filename, 'w-') as f:

        for i in range(len(names)):
            
            temp = K[i][[0,1,0,0,1],[0,1,1,2,2]]
            f.create_dataset(names[i] + r"/matrix", data=temp)
            f.create_dataset(names[i] + r"/rotation", data=r[i])
            f.create_dataset(names[i] + r"/translation", data=t[i])

            
            f.create_dataset("/extrinsic//" + names[i]+ r"/rotation", data=extrinsic_r[i])
            f.create_dataset("/extrinsic//" + names[i]+ r"/translation", data=extrinsic_t[i])

# %%
