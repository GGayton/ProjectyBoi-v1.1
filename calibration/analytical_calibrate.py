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

print("="*50)
print("Using:")
print("Dataset:    ...", dataset)
print("Dots:       ...", dataset_filename[-30:])
print("="*50)

#%% initialise input data structure

calib = AnalyticalCalibration()
inputdata = InputData()

#Load all data
inputdata.load_inputs(dataset_filename)
points,artefact = inputdata.get_inputs_dict()

#Change to homogeneous coords, assume all planes are flat
for ckey in points.keys():
    for pkey in points[ckey].keys():
        points[ckey][pkey] = np.concatenate((points[ckey][pkey], np.ones((points[ckey][pkey].shape[0],1))),axis=1)
        artefact[ckey][pkey][:,2] = 1

#%% Obtain analytical calibration
num_cameras = len(points)
K = {}
ext_t = {}
ext_r = {}

for ckey in points.keys():

    #Invert the projector fy
    if ckey=="projector":
        inv_fy=True
    else:
        inv_fy=False
    
    #obtain camera guess 
    K[ckey], ext_t[ckey], ext_r[ckey] = calib.calibrate(artefact[ckey], points[ckey], inv_fy=inv_fy)

#%% obtain extrinsics guess
r,t = calib.estimate_extrinsics(ext_r,ext_t,"camera")
poseIDs = inputdata.get_pose_IDs()

#%% Reprojection errors

#%% save

filename = r"data\\" + dataset + r"\\parameter outputs\approx.hdf5"
with h5py.File(filename, 'w-') as f:

    for name in inputdata.keys:
        
        f.create_dataset(name + r"/matrix", data=K[name])
        f.create_dataset(name + r"/distortion", data=np.random.rand(7)*0.1+0.05)
        f.create_dataset(name + r"/rotation", data=r[name])
        f.create_dataset(name + r"/translation", data=t[name])

        for pose_id in ext_r[name].keys():
            extrinsic_string = name + r"/extrinsic//" + pose_id
            f.create_dataset(extrinsic_string + r"/rotation", data=ext_r[name][pose_id])
            f.create_dataset(extrinsic_string + r"/translation", data=ext_t[name][pose_id])

# %%
