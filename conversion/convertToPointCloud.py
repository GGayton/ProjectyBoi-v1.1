import os
import sys
import h5py
import numpy as np
import cv2
import matplotlib

currentDir = os.getcwd()
cutoff = currentDir.find("ProjectyBoi2001")
assert cutoff!=-1
home = currentDir[:cutoff+16]

if home not in sys.path:sys.path.append(home)

from clib.Model import Measurement
from clib.Decoding import Decode
from clib.FileFinder import recursiveFileFinder
from clib.cv2PixelSelector import defineArea
from clib.consoleOutputs import ProgressBar

#%% Choose measurement
datasetFileStringList = []
while True:
    datasetFileStringList.append(recursiveFileFinder(home + "\\Output Images\\", ".hdf5"))
    
    uinput = input('Another? y/n')
    if uinput == 'y':
        pass
    else:
        break
    
#%% Mask
uinput = input('Crop area? y/n')

def rescale_to_uint8(array, max_value=[]):
    
    if not max_value:
        max_value = np.iinfo(array.dtype).max
    
    array = array.astype(float)
    array = array*255/max_value
    array = array.astype(np.uint8)
    
    return array

if uinput == 'y':
    with h5py.File(datasetFileStringList[0], 'r') as f:
        image = f["00"][()]
        
    res = image.shape
        
    u,v = np.meshgrid(np.linspace(1,res[0],res[0]), np.linspace(1,res[1],res[1]), indexing = 'ij')
    vec = np.concatenate((u.astype(np.uint16).reshape(-1,1), v.astype(np.uint16).reshape(-1,1)), axis=1)
    # function to display the coordinates of 
    # of the points clicked on the image  
    coords = []
    def click_event(event, x, y, flags, params): 
        global coords
        # checking for left mouse clicks 
        if event == cv2.EVENT_LBUTTONDOWN: 
      
            # displaying the coordinates 
            # on the Shell 
            print("X: {}, Y: {}".format(6*y,6*x))
            coords.append((6*y,6*x))
    
    test_image = rescale_to_uint8(image, max_value=1023)
    
    width = int(test_image.shape[0]/6)
    height = int(test_image.shape[1]/6)
    test_image = cv2.resize(test_image, (width, height))
    
    cv2.imshow('Gamma test', test_image)
    # setting mouse hadler for the image 
    # and calling the click_event() function 
    cv2.setMouseCallback('Gamma test', click_event) 
    cv2.waitKey(0)  
      
    #closing all open windows  
    cv2.destroyAllWindows()
    
    p = matplotlib.path.Path(coords)
    mask = p.contains_points(vec)
    
    mask = mask.reshape(res[0], res[1])

else:
    mask = np.ones_like(image, dtype = bool)
#%% Choose model
parameterFilename = home + "Conversion Script//Parameters//modelParameters.hdf5"

#%% Convert decoded images to 3D points

system = Measurement('Y')
system.inputP.loadParameters(parameterFilename)

decode = Decode()
freq = [1/19, 1/21, 1/23]

xyz = []
for dataset in datasetFileStringList:
    decodedX = decode.modifiedHeterodyne3step(dataset,freq,offsetIndex=1)
    decodedY = decode.modifiedHeterodyne3step(dataset,freq,offsetIndex=10)

    system.inputM.setXMeasurement(decodedX)
    system.inputM.setYMeasurement(decodedY)
    system.inputM.updateMapIndex(mask)
    
    xyz += system.getPointCloud()

fig = matplotlib.pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xyz[0][::1003,0],xyz[0][::1003,1],xyz[0][::1003,2])

#%% Save to notepad file

bar = ProgressBar()

bar.updateBar(0,len(xyz))

for i in range(len(xyz)):
    string = home + \
        "Output PointClouds//" + \
        datasetFileStringList[i][-24:-14] + "-" +\
        datasetFileStringList[i][-13:-5] +".txt"
    np.savetxt(string, xyz[i])
    bar.updateBar(i,len(xyz))
