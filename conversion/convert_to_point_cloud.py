import sys
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt

if "..\\" not in sys.path:sys.path.append("..\\")

from clib.model import Measurement
from clib.decoding import Decode
from commonlib.directory import recursive_file_list_finder
from commonlib.pixel_selector import define_area_ocv as define_area
from commonlib.plotting import plot_pointcloud

#%% Choose measurement
measurement_dir = "..\\measurement images"
filename_list = recursive_file_list_finder(measurement_dir, ".hdf5")
    
#%% Mask

with h5py.File(filename_list[0], 'r') as f:
    image = f["00"][()]
uinput = input('Crop area? y/n')
if uinput == 'y':mask = define_area(image)
else: mask = np.ones_like(image, dtype = bool)
#%% Choose model
parameterFilename = "parameters//model_parameters.hdf5"

#%% Convert decoded images to 3D points

system = Measurement('Y')
system.inputP.load_parameters(parameterFilename)

decode = Decode()
freq = [1/19, 1/21, 1/23]

xyz = []
for dataset in filename_list:
    decodedX = decode.modifiedHeterodyne3step(dataset,freq,offsetIndex=1)
    decodedY = decode.modifiedHeterodyne3step(dataset,freq,offsetIndex=10)

    system.inputM.setXMeasurement(decodedX)
    system.inputM.setYMeasurement(decodedY)
    system.inputM.updateMapIndex(mask)
    
    xyz += system.getPointCloud()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xyz[0][::1003,0],xyz[0][::1003,1],xyz[0][::1003,2], s=5, c=xyz[0][::1003,2])

plot_pointcloud(xyz)

#%% Save to notepad file

# The commented code below will output the point-cloud to a text file - which
# is extremely slow. It is far quicker, and consumes less memory to use specific
# libraries and write to a .hdf5 file or perhaps a .ply file.

# bar = ProgressBar()

# bar.updateBar(0,len(xyz))

# for i in range(len(xyz)):
#     string = home + \
#         "Output PointClouds//" + \
#         datasetFileStringList[i][-24:-14] + "-" +\
#         datasetFileStringList[i][-13:-5] +".txt"
#     np.savetxt(string, xyz[i])
#     bar.updateBar(i,len(xyz))
