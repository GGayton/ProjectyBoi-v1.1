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
params = inputdata.get_serial_estimate()

#%% Initialise
calib = SerialCalibration()

#%% set hyper parameters

calib.options["damping_factor"] = 10
calib.options["verbosity"] = 0
calib.options["max_iterations"] = 20
calib.options["min_change"] = 0.0001
calib.options["max_failure"] = 5

print("="*50)
print("Hyper parameters:")
print("options")
print("="*50)


#%%

test = points["camera"] - calib.transform_TF(artefact["camera"], params["camera"])
#%% clib

L = 0.95
H = 1.05

optimCamParams, Jc, residualsC = calib.train(
    artefact["camera"],
    points["camera"],
    params["camera"],# * tf.random.uniform(initCamParams.shape, minval = L, maxval = H, dtype = DATATYPE),
    )

# %%
