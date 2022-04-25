import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py

currentDir = os.getcwd()
cutoff = currentDir.find("ProjectyBoi2001")
assert cutoff!=-1
home = currentDir[:cutoff+16]
sys.path.append(home)

sys.path.append(home+"Conversion Script")

from clib.FileFinder import recursiveFileFinder
# from clib.Decode miport Decode

#%%
measurement = recursiveFileFinder(home + "Output Images", ".hdf5")

#%%

images = h5py.File(measurement, 'r')

num_of_images = len(images)

images_per_fig = 4

num_of_figs = num_of_images//images_per_fig

remaining_images = num_of_images%images_per_fig

fig_ind = [(0,0), (0,1), (1,0), (1,1)]

i = 0
for k in range(0, num_of_figs):
    
    plt.figure(k)
    
    f, axarr = plt.subplots(2,2) 
    
    for n in range(0, images_per_fig):
        
        #Load array
        array = images["{:02d}".format(i)]
        
        #Iterate
        i = i+1
        
        #Show
        axarr[fig_ind[n]].imshow(array[::6,::6])

if remaining_images:
    k = k+1
    plt.figure(k)
    f, axarr = plt.subplots(2,2) 
    
    for p in range(0, remaining_images):
            
        #Load array
        array = images["{:02d}".format(i)]
        
        #Iterate
        i = i+1
            
        #Show
        axarr[fig_ind[p]].imshow(array)

images.close()
    
   


