
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py

currentDir = os.getcwd()
cutoff = currentDir.find("ProjectyBoy2000")
assert cutoff!=-1
home = currentDir[:cutoff+16]

if home not in sys.path: sys.path.append(home)
if home+"Calibration" not in sys.path: sys.path.append(home+"Calibration")
if home+"Calibration\Dot localisation v2" not in sys.path: sys.path.append(home+"Calibration\Dot localisation v2")
if home+"Phase decoding" not in sys.path:sys.path.append(home+"Phase decoding")

from DotLocalisation import DotLocalisation

from Decoding import Decode

from commonlib.h5py_functions import load_h5py_arrays, tryDel, num_of_keys

from commonlib.common_functions import listdir

from scipy.stats import lognorm


    
#%% Load data

dataset = "2022_02_24"

rawDataDir = home + r"\Calibration\Data\\" + dataset + "\\Raw Data\\"
rawDataList = listdir(rawDataDir, lambda x: x[-5:]=='.hdf5')

phaseMapDataset = home + r"\Calibration\Data\\" + dataset + "\\Inputs\\phaseMaps.hdf5"

dotDir = home + r"\Calibration\Data\\" + dataset + "\\Inputs\\dots2.hdf5"

num_of_positions = num_of_keys(dotDir, "camera/points")

cPoints = []
cA=[]
cB=[]
cT=[]
cV=[]

pPoints = []
pV=[]
        
with h5py.File(dotDir, 'r') as f:
    
    ones = np.ones((184,1))
    
    for i in range(0,num_of_positions):
        
        string = "{:02d}".format(i)
                        
        cPoints.append(
            np.concatenate((f["camera/points"][string][:,:].astype(np.float64), ones), axis=1))
        pPoints.append(
            np.concatenate((f["projector/points"][string][:,:].astype(np.float64), ones), axis=1))
        
        cA.append(f["camera/A"][string][()])
        cB.append(f["camera/B"][string][()])
        cT.append(f["camera/theta"][string][()])
                
        cV.append(f["camera/covariance"][string][()])
        pV.append(f["projector/covariance"][string][()])
        
analysisFilename = home+r"Calibration\Reproducibility\dotAnalysisOut.hdf5"

with h5py.File(analysisFilename, 'r') as f:
    
    insidePDF = []
    outsidePDF= []
    
    insideHist = []
    outsideHist = []
    
    speckleList = []
    speckleMaxList = []
    
    sigmaList = []
    sigmaStdList = []
    
    for i in range(25):
        kString = "{:02d}".format(i)
        
        insidePDF.append(f[kString + r"//insidePDF"][()])
        outsidePDF.append(f[kString + r"//outsidePDF"][()])

        insideHist.append(f[kString + r"//insideHist"][()])
        outsideHist.append(f[kString + r"//outsideHist"][()])
        
        speckleList.append(f[kString + r"//speckle"][()])
        speckleMaxList.append(f[kString + r"//speckleMax"][()])
        
        sigmaList.append(f[kString + r"//res"][()])
        sigmaStdList.append(f[kString + r"//resSTD"][()])
        
#%% Adapt poor PDF results

for i in range(len(insidePDF)):
    
    for j in range(len(insidePDF[i])):
        
        if np.isnan(insidePDF[i][j,0]):
            
            insidePDF[i][j,0] = 0.5
            insidePDF[i][j,2] = 1
        
        elif insidePDF[i][j,0]==0:
            
            insidePDF[i][j,0] = 0.5
            insidePDF[i][j,2] = 1
            
#%% Clean sigma results

for i in range(len(sigmaList)):
    
    for j in range(len(sigmaList[i])):
        
        if sigmaList[i][j]>3.5:
            
            sigmaList[i][j] = 3.5
#%% Clean sigma std results

for i in range(len(sigmaList)):
    
    for j in range(len(sigmaList[i])):
        
        if sigmaStdList[i][j]>0.5:
            
            sigmaStdList[i][j] = 0.5

#%% Initialise

localiser = DotLocalisation()
#%% Base worst on cV

w = []
for i in range(num_of_positions):

    w.append(np.argmax(cV[i][:,0,0]))
#%% Base worst on pV
w = []
for i in range(num_of_positions):

    w.append(np.argmax(pV[i][:,0,0]))

#%% Base worst on worst contrast
w = []
for i in range(len(insidePDF)):
    temp = []
    for j in range(len(insidePDF[i])):
        
        temp1,_ = lognorm.stats(insidePDF[i][j][0], insidePDF[i][j][1],insidePDF[i][j][2], moments='mv')
        temp2,_ = lognorm.stats(outsidePDF[i][j][0], outsidePDF[i][j][1],outsidePDF[i][j][2], moments='mv')
        
        temp.append(temp2-temp1)
    w.append(temp)

del temp, temp1, temp2

for i in range(len(w)):
    w[i] = np.argmin(w[i])
    
#%% Base worst on largest sigma
w = []
for i in range(len(insidePDF)):
    w.append(np.argmax(sigmaList[i]))
    
#%% Base worst on largest sigma
w = []
for i in range(len(insidePDF)):
    w.append(np.argmax(sigmaStdList[i]))
#%% Base worst on speckle
w = []
for i in range(len(insidePDF)):
    w.append(np.argmax(speckleList[i]))
    
#%% Base worst on speckle max
w = []
for i in range(len(insidePDF)):
    w.append(np.argmax(speckleMaxList[i]))
#%% troubleshoot all
plt.close('all')
for i in range(10):#num_of_positions):
    
    datasetString = rawDataDir + rawDataList[i]    
    blankImage = load_h5py_arrays(datasetString, 0)
    
    print('== {:02d} =='.format(i))
    localiser.troubleshootLocalisation(blankImage, w[i])
    
#%% trouble shoot specific
N = len(cPoints)

k = np.random.randint(0,N)
j = np.random.randint(0,184)

k = 12

decodeType = 'filtered'

print('== {:02d} =='.format(k))

datasetString = rawDataDir + rawDataList[i]    
blankImage = load_h5py_arrays(datasetString, 0)

# localiser.troubleshootLocalisation(blankImage, w[i])

with h5py.File(phaseMapDataset, 'r') as f:
    mappingX = f[decodeType+"//{:02d}//X".format(k)][()]
    mappingY = f[decodeType+"//{:02d}//Y".format(k)][()]
    
params = np.empty((184,5))
params[:,:2] = cPoints[k][:,:2]
params[:,2] = cA[k]
params[:,3] = cB[k]
params[:,4] = cT[k]  
 
localiser.troubleshootInference(params,mappingX,mappingY,j)

#%% troubleshoot specific

j = np.random.randint(0,184)
 
localiser.troubleshootInference(params,mappingX,mappingY,j)

