from PySide2.QtCore import (
    QMutex,
    QTimer,
    QObject,
    Signal,
    Slot,
    )

from PySide2.QtGui import (
    QPixmap,
    QImage,
    QColor, 
    qRgb
    )
from mlib.common_functions import(
    convert_float_to_uint8_image,
    )

from mlib.common_qt_functions import(
    array_to_QPixmap
    )

from mlib.console_outputs import (ProgressBar)

import os
import numpy as np
import time
import h5py

class TaskManager(QObject):
    
    """
    This class controls the measurement sequence (loading of images/taking of images).
    
    The inputs are the 
    
    QSemaphore object: finish_cond (Tells the measurement class all is finished)
    Queue object : projector_queue (which passes images to the projector after laoding them)
    QSemaphore object: projector_cond (Tells the projector to project an image)
    QSemaphore object: camera_cond (Tells the camera to take an image)
        
    There is some deprecated code in here too.
    
    main functions here are measure() and load_in_images()
    """
    
    project_N_images_signal = Signal(int)
    acquire_N_images_signal = Signal(int)
    save_N_images_signal = Signal(int)
    
    toggle_camera_auto_mode_signal = Signal()
    command_signal = Signal()
    
    
    def __init__(self, finish_cond, projector_queue, projector_cond, camera_cond):
        super().__init__()
        
        self.projector_queue = projector_queue
        self.camera_cond = camera_cond
        self.projector_cond = projector_cond
        self.finish_cond = finish_cond
        
        self.N = None
        
        self.image_stack = None
        
        self.nth_image = 0
                
        # self.minVal = np.nonzero(self.gammaLUT)[0][0]
        # self.maxVal = np.nonzero(self.gammaLUT)[0][-1]
        self.minVal = 0
        self.maxVal = 0
        
        self.minVal = 0
        self.maxVal = 255
        
    @Slot(str)
    def load_in_images(self, directory):       
    
        #Preallocate
        self.image_stack = []
        
        bar = ProgressBar()
        
        i=0
        #Load in files
        with h5py.File(directory, 'r') as f:
            self.N = len(list(f.keys()))
            for key in f.keys():
                
                #Load in the numpy array
                temp_image = f[key][()]

                #Convert numpy array to uint8
                temp_image = convert_float_to_uint8_image(temp_image, self.minVal, self.maxVal)
                
                #Convert numpy array to QPixmap
                self.image_stack.append(array_to_QPixmap(temp_image, image_type = 'Grayscale8'))
                
                bar.updateBar(i,self.N-1)
                i=i+1

        print('Loaded {} images.'.format(i+1))
        
    def pass_to_projector(self, N):
        
        for i in range(0, N):
            self.projector_queue.put(self.image_stack[i])
    
    @Slot()
    def project_nth_image(self):
        
        self.projector_queue.put(self.image_stack[self.nth_image])
        
        self.project_N_images_signal.emit(1)
        
        self.camera_cond.acquire(1)
    
    @Slot()
    def set_nth_image(self, N):
        
        self.nth_image = N
    
    @Slot()
    def measurement(self):
        
        t1 = time.time()
        print("==== [Measurement beginning...] ====")
                
        self.measure(self.N)
        
        #Disconnect
        print("==== [Complete in {}secs] ====".format(np.round(time.time()-t1, 2)))
                                       
    def measure(self, N):
        #Initialise the projector to project N pictures
        self.project_N_images_signal.emit(N)
        
        #Initialise the camera to take N pictures
        self.acquire_N_images_signal.emit(N)
        
        #Initialise the image saver to save N pictures
        self.save_N_images_signal.emit(N)
                
        #Pass the images to the projector
        self.pass_to_projector(N)
        
        #Wait for finish condition
        self.finish_cond.acquire(3)
        
        #Clear the last camera condition            
        self.projector_cond.acquire(1)

