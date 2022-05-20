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

from calib.input_data import InputData
from calib.serial import SerialCalibration
from calib.analytical_calib import AnalyticalCalibration

#%% set datasets
dataset = "2022_02_24"
dataset_filename =  r"data\\" + dataset + r"\\Inputs\dotsTest.hdf5"
estimate_filename = r"data\\" + dataset + r"\\Parameter outputs\\approx.hdf5"

print("="*50)
print("Using:")
print("Dataset:    ...", dataset)
print("Dots:       ...", dataset_filename[-30:])
print("Estimate:   ...", estimate_filename[-30:])
print("="*50)

#%% initialise input data structure

inputdata = InputData()

#Load all data
inputdata.load_inputs(dataset_filename)
inputdata.load_estimate(estimate_filename)

points,artefact = inputdata.get_inputs_TF()

# inputdata.params["camera"]["distortion"] = np.zeros(7)
params = inputdata.get_serial_estimate()

#%% Initialise
calib = SerialCalibration()

#%% set hyper parameters

calib.options["damping_factor"] = 10
calib.options["verbosity"] = 1
calib.options["max_iterations"] = 200
calib.options["min_change"] = 0.0001
calib.options["max_failure"] = 5

print("="*50)
print("Hyper parameters:")
print(calib.options)
print("="*50)

#%%
J = calib.jacobian(artefact["camera"], params["camera"])

#%% clib

L = 0.95
H = 1.05

optim_params = {}
J = {}
residuals = {}

for key in inputdata.keys:
    optim_params[key], J[key], residuals[key] = calib.train(
        artefact[key],
        points[key],
        params[key],# * tf.random.uniform(params[key].shape, minval = L, maxval = H, dtype = DATATYPE),
        )
# %% reprojection errors

plt.close('all')
for key in inputdata.keys:
    plt.figure()
    plt.scatter(residuals[key][::2].numpy(), residuals[key][1::2], s=3)
    print(key, ": ", np.sum(residuals[key].numpy()**2)**0.5)

#%% assemble parameters

K,D,ext_r,ext_t = {},{},{},{}
for key in inputdata.keys:
    K[key],D[key],ext_r[key],ext_t[key] =\
        calib.assemble_parameters(artefact[key], optim_params[key])

#%% estimate extrinsics
ecalib = AnalyticalCalibration()
r,t = ecalib.estimate_extrinsics(ext_r,ext_t,"camera")

#%% save

filename = r"data\\" + dataset + r"\\parameter outputs\serial_parameters.hdf5"
with h5py.File(filename, 'w-') as f:

    for name in inputdata.keys:
        
        f.create_dataset(name + r"/matrix", data=K[name])
        f.create_dataset(name + r"/distortion", data=D[name])
        f.create_dataset(name + r"/rotation", data=r[name])
        f.create_dataset(name + r"/translation", data=t[name])

        for pose_id in ext_r[name].keys():
            extrinsic_string = name + r"/extrinsic//" + pose_id
            f.create_dataset(extrinsic_string + r"/rotation", data=ext_r[name][pose_id])
            f.create_dataset(extrinsic_string + r"/translation", data=ext_t[name][pose_id])

# %%
