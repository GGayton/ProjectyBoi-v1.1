#%% imports
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py

home = "..\\"
if home not in sys.path: sys.path.append(home)

from commonlib.directory import recursive_file_finder

measurement = recursive_file_finder(home + "measurement images", ".hdf5")

#%% plot all images

with h5py.File(measurement, 'r') as images:

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
            axarr[fig_ind[p]].imshow(array[::6,::6])
    

    
   


