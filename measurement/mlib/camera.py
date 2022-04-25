from vimba import *
from PySide2.QtCore import (
    QSemaphore,
    QMutex,
    QObject,
    Signal,
    Slot,
    )
from PySide2.QtWidgets import (
    QApplication
    )
import time

class CameraControl(QObject):
    
    """
    This class controls the camera.
    
    The inputs are the 
    
    Queue object : camera_queue (which passes the camera images to the saver thread)
    QSemaphore object: camera_cond (Tells the camera to take an image)
    QSemaphore object: projector_cond (Tells the projector to project an image)
    QSemaphore object: finish_cond (Tells the measurement class all is finished)
    
    It contains the function to set the camera to the correct settings, 
    which is done before a meausurement, the function to take images when "streaming" 
    (the images taken before starting a measurement to optimise aperture size etc), and the
    function for starting the measurement.
    
    During measurement, the camera waits for a signal from a QSemaphore object
    
    Almost all of the code in here is taken from the vimba code manual - which is a mnaual for controlling Prosilica GT cameras with python.
    """
        
    def __init__(self, camera_queue, camera_cond, projector_cond, finish_cond):
        super().__init__()
        
        #Queue for streaming frames through
        self.camera_queue = camera_queue
        
        #Wait condition
        self.camera_cond = camera_cond
        self.projector_cond = projector_cond
        self.finish_cond = finish_cond
        
        #Operating mode
        self.continuous_mode = True
        self.repeat_N = 1

        self.measurement_settings = {
            "ExposureMode": "Timed",
            "TriggerActivation": "RisingEdge", 
            "TriggerMode": "On",
            "TriggerOverlap": "Off",
            "TriggerSource": "Line1",
            "TriggerSelector": "FrameStart",
            "TriggerDelayAbs": 0,
            
            "DecimationHorizontal": 1,
            "DecimationVertical": 1,
            "BinningHorizontal": 1,
            "BinningVertical": 1,
            "ExposureTimeAbs": 8970,
            "Height": 5120,
            "Width": 5120,
            "AcquisitionMode": "SingleFrame"
            }
                           
    def setup_camera(self, cam: Camera):
        with cam:
            # Try to adjust GeV packet size. This Feature is only available for GigE - Cameras.
            try:
                cam.GVSPAdjustPacketSize.run()
        
                while not cam.GVSPAdjustPacketSize.is_done():
                    pass
        
            except (AttributeError, VimbaFeatureError):
                pass
            
    def get_camera(self, camera_id):
        with Vimba.get_instance() as vimba:
            if camera_id:
                try:
                    return vimba.get_camera_by_id(camera_id)
    
                except VimbaCameraError:
                    raise Exception('Failed to access Camera \'{}\'. Abort.'.format(camera_id))
    
            else:
                cams = vimba.get_all_cameras()
                if not cams:
                    raise Exception('No Cameras accessible. Abort.')
    
                return cams[0]
                
    def set_mode(self, cam, setting_list):
        
        #Set settings
        for setting in setting_list:
            attribute = getattr(cam, setting)
            attribute.set(setting_list[setting])
                                             
    @Slot()
    def start_streaming(self):
        
        cam_id = None
        
        print("Camera:         Images now acquired on demand")
    
        with Vimba.get_instance():
            with self.get_camera(cam_id) as cam:

                #Set the streaming parameters
                self.set_mode(cam, self.measurement_settings)

                #Set pixels to mono8
                cam.set_pixel_format(PixelFormat.Mono10)

                #Set the correct values for GIGE
                self.setup_camera(cam)
                
                self.streaming = True
                
                while self.streaming:
                    
                    #Wait for signal
                    condition = self.camera_cond.tryAcquire(1, timeout = 1000)
                    
                    if condition:
                                        
                        try:
                            #Acquire frame                    
                            frame = cam.get_frame(timeout_ms=6000)
                        except Exception as e:
                            print("Camera:        ", e)
                            
                        print("Camera:         Acquired image")
                        
                        #Convert frame
                        frame.convert_pixel_format(PixelFormat.Mono16)
                        pix_array = frame.as_opencv_image()
                        
                        #Pass to saver
                        self.camera_queue.put(pix_array[:,:,0])
                            

                        
                    QApplication.processEvents()
                    
        print("Camera:         COMPLETE")
                                      
    @Slot(int)
    def take_N_images(self, N):
                
        cam_id = None
        
        print("Camera:         Starting measurement for {} image(s)".format(N))
    
        with Vimba.get_instance():
            with self.get_camera(cam_id) as cam:

                #Set the streaming parameters
                self.set_mode(cam, self.measurement_settings)

                #Set pixels to mono8
                cam.set_pixel_format(PixelFormat.Mono10)

                #Set the correct values for GIGE
                self.setup_camera(cam)
                
                for i in range(0,N):
                    
                    #Wait for signal
                    self.camera_cond.acquire(1)
                                                            
                    for n in range(0, self.repeat_N):
                        
                        #Repeat measurements that fail
                        while True:
                            try:
                                #Acquire frame
                                time.sleep(0.3)
                                frame = cam.get_frame(timeout_ms=2000)
                                
                            except Exception as e:
                                print("Camera:        ", e)
                                print("Camera:         Retrying...")
                                
                            break
                            
                        print("Camera:         Acquired image {:02d}".format(i))
                        print(frame)
                        
                        #Convert frame
                        frame.convert_pixel_format(PixelFormat.Mono16)
                        pix_array = frame.as_opencv_image()
                        
                        #Pass to saver
                        self.camera_queue.put(pix_array[:,:,0])
                        time.sleep(0.3)
                        
                    #Signal Projector
                    if self.continuous_mode:
                        self.projector_cond.release(1)
                    
                            
        print("Camera:         COMPLETE")
        self.finish_cond.release(1)
        
        self.auto_mode = not self.auto_mode
        print("Camera:         Auto mode is {}".format(self.auto_mode))
   
    @Slot()
    def stop_streaming(self):
       
       self.streaming=False
   
    @Slot()
    def toggle_continuous_mode(self):
        
        self.continuous_mode = not self.continuous_mode
        print("Camera:         Continuous mode is {}".format(self.continuous_mode))
        
    @Slot(int)
    def set_repeat_images(self, N):
        
        self.repeat_N = N
        print("Camera:         Taking {} camera image(s) for every projector image".format(self.repeat_N))
        
    @Slot()
    def clear_camera_queue(self):

        try:
            queue_size = self.camera_queue.qsize()
        except Exception as e:
            print(e)

        for item in range(0, queue_size):

            self.camera_queue.get_nowait()
            
        print("Camera:         Queue is cleared, {} image(s) remain in the queue".format(self.camera_queue.qsize()))
        
    @Slot()
    def clear_camera_cond(self):

        out = self.camera_cond.tryAcquire(self.camera_cond.available(), 1000)

        if out:
            print("Camera:         Condition is cleared, {} remaining pictures to be taken".format(self.camera_cond.available()))
        else:
            print("Camera:         Condition is clear failed, {} remaining pictures to be taken".format(self.camera_cond.available()))
                