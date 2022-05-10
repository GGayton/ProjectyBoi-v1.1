import sympy as sp
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

from commonlib.h5py_functions import load_h5py_arrays, tryDel

from commonlib.common_functions import listdir

from commonlib.common_functions import (
    sampleImageGradient,
    extractRegion,
    interp2D, 
    rescale_to_uint8,
    corrMatrixfromCovMatrix
    )
    
#%% Choose datasets

dataset = "2021_12_20"

datasetDir = home + r"\Calibration\Data\\" + dataset + "\\Raw Data\\"
datasetList = listdir(datasetDir, lambda x: x[-5:] == ".hdf5")

phaseMapDataset = home + r"\Calibration\Data\\" + dataset + "\\Inputs\\phaseMaps.hdf5"



dotFilename = home + r"\Calibration\Data\\" + dataset + "\\Inputs\\dots3.hdf5"

#%% Load data

cParams = []
cV = []
with h5py.File(dotFilename, 'r') as f:
    nPos = len(list(f["camera//points"].keys()))
    for i in range(0,nPos):
        
        string = "{:02d}".format(i)
        
        points = f["camera//points//{:02d}".format(i)][()].reshape(-1,2)
        A = f["camera//A//{:02d}".format(i)][()].reshape(-1,1)
        B = f["camera//B//{:02d}".format(i)][()].reshape(-1,1)
        T = f["camera//theta//{:02d}".format(i)][()].reshape(-1,1)
        
        cParams.append(np.concatenate((points, A, B, T),axis=1))
        
        cV.append(f["camera//covariance//{:02d}".format(i)][()])
        
        
#%% Initialise
localiser = DotLocalisation()

#%% define jacobian

def defineEllipseParamJacobian():
    x,y,x0,y0,A,B,T = sp.symbols("x,y,x0,y0,A,B,T", real = True)
    
    f = ((x-x0)*sp.cos(T) - (y-y0)*sp.sin(T))**2/A**2 + ((x-x0)*sp.sin(T) + (y-y0)*sp.cos(T))**2/B**2 - 1

    w = [x0,y0,A,B,T]
    
    J = sp.Matrix([[0,0,0,0,0]])
    for j in range(len(w)):
    
        J[0,j] = f.diff(w[j])
    
    return sp.lambdify([x,y,x0,y0,A,B,T], J)

def plotEllipse(params):
    
    x0,y0,a,b,T = params
    
    x,y = np.meshgrid(np.linspace(x0//1 - 50, x0//1 + 50,101), np.linspace(y0//1 - 50, y0//1 + 50,101), indexing = 'ij')
    
    f = ((x-x0)*np.cos(T) - (y-y0)*np.sin(T))**2/a**2 + ((x-x0)*np.sin(T) + (y-y0)*np.cos(T))**2/b**2 - 1
    
    plt.contour(f, [0])

#%%
plt.close('all')
k = np.random.randint(0,25)
i = np.random.randint(184)

k=16
j=18

with h5py.File(datasetDir + datasetList[k], 'r') as f:
        blankImage = f["00"][()]

dot = cParams[k][j,:2]

array,_,_ = extractRegion(blankImage, dot, 50)


plt.imshow(array)
# plt.figure()
plotEllipse(cParams[k][j,:])

#%%

Jfunc = defineEllipseParamJacobian()

x0,y0,a,b,T = cParams[k][j,:]
x,y = np.meshgrid(np.linspace(x0//1 - 50, x0//1 + 50,1001), np.linspace(y0//1 - 50, y0//1 + 50,1001), indexing = 'ij')

f = ((x-x0)*np.cos(T) - (y-y0)*np.sin(T))**2/a**2 + ((x-x0)*np.sin(T) + (y-y0)*np.cos(T))**2/b**2 - 1

J = Jfunc(x,y,x0,y0,a,b,T)
J = J[0,...]

V = cV[k][j,...]*1

s = np.empty_like(f)

for n in range(s.shape[0]):
    for m in range(s.shape[1]):
        
        s[n,m] = J[:,n,m].reshape(1,-1) @ V @ J[:,n,m].reshape(-1,1)
        
plt.imshow(np.abs(f)<2*s**0.5)
#%%

decodeType = 'filtered'
print("Decoding as ", decodeType)

with h5py.File(phaseMapDataset, 'r') as f:
        mappingX = f[decodeType+"//{:02d}//X".format(k)][()]
        mappingY = f[decodeType+"//{:02d}//Y".format(k)][()]    

datasetString = datasetDir + datasetList[k]    

blankImage = load_h5py_arrays(datasetString, 0)



