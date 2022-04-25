import datetime
import os
import numpy as np
import time
import h5py

from PySide2.QtCore import (
    QObject,
    Slot
    )

class ImageSaver(QObject):
    
    """
    This class saves all the images
    
    The inputs are the 
    
    Queue object : camera_queue (which passes the camera images to the saver thread)
    QSemaphore object: finish_cond (Tells the measurement class all is finished)
    string object: gives the directory to save in
        
    """
    
    def __init__(self, camera_queue, finish_cond, save_directory=None):
        super().__init__()
        
        self.finish_cond = finish_cond
        
        self.camera_queue = camera_queue
        
        self.save_directory = save_directory
        
        self.save_name = ""
        
        self.save_name = None
        
        self.repeat_N = 1

    def clear_queue(self):
        
        clearing = True
        time.sleep(1)
        while clearing:
            
                if self.camera_queue.empty():
                    clearing = False
                else:
                    self.camera_queue.get()
                    time.sleep(0.1)
         
        print("Saver:          Queue cleared")
        
    @Slot(int)
    def set_repeat_images(self, N):
        
        self.repeat_N = N
        
        print("Saver:          Taking {} repeat image(s)".format(N))

#%% Save measurements        
    def create_measurement_directory(self):
        #Find time
        today = datetime.datetime.now()
        
        #Create directory based on date 
        date_directory = self.save_directory + '\\' + str(today.date())
        
        try:
            os.mkdir(date_directory)
        except:
            pass
        
        #Create directory based on time
        time_directory = date_directory + "\\{:02d}-{:02d}-{:02d}".format(today.hour,today.minute,today.second)
                            
        return time_directory
    
    def increment_mean(self, oldMean, newObs, N):
        return oldMean + (newObs - oldMean)/N

    @Slot(int)
    def save_N_images(self, N):
        
        print("Saver:          Starting saving {} image(s)".format(N))

        measurement_directory = self.create_measurement_directory()
        
        #Try writing h5py file
        try:
            f = h5py.File(measurement_directory + ".hdf5", 'w-')

            for i in range(0,N):
    
                pix_array = np.zeros((5120,5120), dtype = float)
                
                for j in range(0,self.repeat_N):
                    
                    #Get from queue
                    pix_array = self.increment_mean(pix_array, self.camera_queue.get(timeout=10), j+1)
                    
                #Check
                if pix_array.max()<400:
                    print("Saver:          POSSIBLE FAILURE ON IMAGE {:02d}".format(i))
                    
                #Store in file
                f.create_dataset("{:02d}".format(i),data=np.round(pix_array).astype(np.uint16), compression="lzf")
                
                print("Saver:          Saved image {:02d}".format(i))
                
        except Exception as e:
            print("Saver:          FAILURE", e)
        
        finally:
            f.close()    
        
        #Check
        try:
            f = h5py.File(measurement_directory+".hdf5", 'r')
            
            if len(f)!=N:
                print("Saver:          FAILURE")
            elif len(f)==N:
                print("Saver:          COMPLETE") 
                
        except Exception as e:
            print("Saver:          FAILURE:", e)
         
        self.finish_cond.release(1)
    
    @Slot(int)
    def bin_N_images(self, N):
        
        for i in range(0,N):

            self.camera_queue.get(timeout=10)
            
        self.finish_cond.release(1)
#%% Save information
    def save_information(self, save_directory):
        
        saving = True
        
        while saving:
            
            pointer = self.camera_queue.get()
            
            #If LUT 
            if pointer == "LUT":
                print("Saver:          Writing LUT") 
                LUT = self.camera_queue.get()
                self.store_LUT(LUT, save_directory)
             
            #Regime writing
            elif pointer == "REGIME":
                print("Saver:          Writing regime") 
                regime_list = self.camera_queue.get()

                self.store_regime_list(regime_list, save_directory)
        
            #Exit condition
            elif pointer == "0":
                saving = False
            
            else:
                print("Saver:          What is this?")
                print(pointer)
            
            print("Saver:          Complete")    
                 
    def store_LUT(self, numpy_array, save_directory):
        
        LUT_filename = save_directory + "LUT"
        
        try:
            np.save(LUT_filename, numpy_array)
        except:
            print("Saver:          LUT save failed")
        


        