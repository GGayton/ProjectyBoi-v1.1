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

from commonlib.common_functions import listdir, rodrigues

from scipy.stats import lognorm


    
#%% Load data

dataset = "2022_02_24"

rawDataDir = home + r"\Calibration\Data\\" + dataset + "\\Raw Data\\"
rawDataList = listdir(rawDataDir, lambda x: x[-5:]=='.hdf5')

paramDir = home  + "\Calibration\Data\\" + dataset + "\\Parameter outputs\\modelFW_seed.hdf5"

boarDir = home  + "\Calibration\Data\\" + dataset + "\\Inputs\\board.hdf5"

phaseMapDataset = home + r"\Calibration\Data\\" + dataset + "\\Inputs\\phaseMaps.hdf5"

dotDir = home + r"\Calibration\Data\\" + dataset + "\\Inputs\\dots.hdf5"

num_of_positions = num_of_keys(dotDir, "camera/points")

cPoints = []
cA=[]
cB=[]
cT=[]
cV=[]
cS=[]

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
        cS.append(f["/camera/sigma/"][string][()])
        cV.append(f["camera/covariance"][string][()])
        pV.append(f["projector/covariance"][string][()])
        
with h5py.File(paramDir, 'r') as f:
    
    cArray = f["camera/array"][()]
    pArray = f["projector/array"][()]
    
with h5py.File(boarDir, 'r') as f:
    
    board = f["board"][()]
cS = np.hstack(cS).reshape(-1,1)
#%% 
nPos = (cArray.shape[0] - 12)//6

R = []
T = []
for i in range(nPos):
    
    R += [rodrigues(cArray[12 + 6*i: 15 + 6*i])]
    T += [cArray[15 + 6*i: 18 + 6*i].reshape(1,-1)]

x = []
for i in range(nPos):
    x += [board@R[i].T + T[i]]
    
x = np.vstack(x)

#%%

plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sact = ax.scatter(x[:,0], x[:,1], x[:,2], c = cS, s=10, cmap = 'inferno')
cbar = plt.colorbar(sact)
cbar.ax.set_ylabel('Line-spread function width / pixels', fontsize = 14)
cbar.ax.tick_params(labelsize = 14)
ax.set_xlabel('X / mm', fontsize = 14, labelpad = 10)
ax.set_ylabel('Y / mm', fontsize = 14, labelpad = 10)
ax.set_zlabel('Z / mm', fontsize = 14, labelpad = 10)

ax.tick_params(labelsize = 14)

fig = plt.figure()
ax = fig.add_subplot(111)
scat = ax.scatter(np.sum(x**2, axis=1)**0.5, cS,c = np.sum(x[:,:2]**2,axis=1)**0.5, s=3, cmap = 'inferno')
cbar = plt.colorbar(scat)
cbar.ax.set_ylabel('Lateral distance / mm', fontsize = 14)
cbar.ax.tick_params(labelsize = 14)
ax.set_xlabel('Distance from origin / mm', fontsize = 14)
ax.set_ylabel('Line-spread function width / pixels', fontsize = 14)

ax.tick_params(labelsize = 14)

#%%
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


x = []
for i in range(nPos):
    x += [board@R[i].T + T[i]]
    
plt.close('all')
plt.figure()
plt.plot(board[:,0], board[:,1], '.')

I = [0,7,15,175,183,176,0]

plt.plot(board[I,0], board[I,1], 'r')

# x = [0,1,1,0]
# y = [0,0,1,4]
# z = [0,1,0,1]
# verts = [list(zip(x,y,z))]

# verts = np.zeros((6,4))
verts = x[0][I]

# verts = np.array([[0,0,0],[1,0,1],[1,1,0],[0,1,1]])
# x=x[0]
# verts = []
# for i in range(len(I)):
#     verts+= [(x[I[i], 0], x[I[i], 1], x[I[i], 2])]
# verts = [verts]
    
    
# plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.add_collection3d(Poly3DCollection(verts, closed=True))

for i in range(len(x)):
    ax.plot_trisurf(x[i][:,0], x[i][:,1], x[i][:,2], alpha=0.2)
    ax.plot(x[i][I,0], x[i][I,1], x[i][I,2], 'k', linewidth=0.5)
# ax.set_xlim([-100,100])
# ax.set_ylim([-100,100])
# ax.set_zlim([300,800])

ax.tick_params(labelsize = 14)
ax.set_xlabel('X / mm', fontsize = 14)
ax.set_ylabel('Y / mm', fontsize = 14)
ax.set_zlabel('Z / mm', fontsize = 14)

