from PySide2.QtCore import (
    QMutex,
    QTimer,
    QObject,
    Signal,
    Slot,
    )

from commonlib.common_functions import convert_float_to_uint8_image
from commonlib.common_qt_functions import array_to_QPixmap
from commonlib.console_outputs import ProgressBar

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
    bin_N_images_signal = Signal(int)

    
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
        
        self.warm_up_N = 10
        
        self.repetitions = 1
        
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
    
    @Slot()
    def repeat_measurement(self):
        t2 = time.time()
        for i in range(self.repetitions):
            t1 = time.time()
            print("==== [Measurement {:02d}...] ====".format(i))
            try:
                self.measure(self.N)
            except Exception as e:
                raise e
            
            print("==== [Measurement {:02d} comlpeted: {}secs] ====".format(i, np.round(time.time()-t1, 2)))
            time.sleep(5)
        
        #Disconnect
        print("==== [Complete in {}secs] ====".format(np.round(time.time()-t2, 2)))
            
    @Slot(int)
    def set_repetitions(self,N):
        
        self.repetitions = N
        print("Measurement:     Taking {} measurement(s)".format(self.repetitions))
    #%%
    @Slot(int)
    def set_warm_up(self,N):
        
        self.warm_up_N = N
        print("Warm-up:         Taking {} image(s)".format(self.warm_up_N))

    
    @Slot()
    def warm_up(self):
        
        t1 = time.time()
        print("==== [Warm-up beginning...] ====")
        
        self.warm_up_ex(self.warm_up_N)
        
        print("==== [Complete in {}secs] ====".format(np.round(time.time()-t1, 2)))
        
    def warm_up_ex(self, N):
        try:
            
            #Pass the images to the projector
            self.create_random_images(N)
            self.pass_to_projector(N)
            
            #Initialise the camera to take N pictures
            self.acquire_N_images_signal.emit(N)
            
            #Initialise the projector to project N pictures
            self.project_N_images_signal.emit(N)
                        
            #Initialise the image saver to save N pictures
            self.bin_N_images_signal.emit(N)
                        
        except Exception as e:
            print(e)

        #Wait for finish condition
        self.finish_cond.acquire(3)
        
        #Clear the last camera condition            
        self.projector_cond.acquire(1)
        
    def create_random_images(self,N):
        
        self.image_stack = []
        bar = ProgressBar()
        bar.updateBar(0,N)

        
        for i in range(N):
            
            temp_image = np.random.randint(0,255,(912,1140)).astype(np.uint8)
                           
            #Convert numpy array to QPixmap
            self.image_stack.append(array_to_QPixmap(temp_image, image_type = 'Grayscale8'))
            
            bar.updateBar(i+1,N)

            