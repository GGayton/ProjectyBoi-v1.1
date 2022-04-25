import os
import time
from PySide2.QtWidgets import (
    QApplication,
    QLabel,
    QWidget,
    )

from PySide2.QtGui import (
    QPixmap,
    QImage
    )

from PySide2.QtCore import (
    Qt,
    QMutex,
    Slot,
    )

class ProjectorControl(QWidget):    

    def __init__(self, projector_queue, projector_cond=None, camera_cond = None, finish_cond=None):
        super().__init__()
        
        self.projector_queue = projector_queue
        self.projector_cond = projector_cond
        self.camera_cond = camera_cond
        self.finish_cond = finish_cond
                
        #Current directory
        self.current_directory = os.path.dirname(os.path.realpath(__file__))
        
        #Load images
        self.home_image = QPixmap("Home_screen.png")
        self.target_image = QPixmap("Target_screen.png")
                      
        #Create the image label
        self.label = QLabel()
        
        #Initialise vairables
        self.image_index = -1
        
        #Get screen information - take projector as screen 2
        screens = QApplication.screens()       
        projector_screen = screens[1]
        
        #Position label to correct monitor           
        self.label.setFixedSize(
            projector_screen.availableGeometry().width(),\
            projector_screen.availableGeometry().height())    
        self.label.move(
            projector_screen.availableGeometry().left(),\
            projector_screen.availableGeometry().top())  
        self.label.setWindowState(Qt.WindowFullScreen)
                
        self.label.show()
        
        self.project_home_screen()

    #Change projector pixels and update     
    def project(self, image):
        
        #set image
        try:
            self.label.setPixmap(image)
        except Exception as e:
            print(e)

        #Update the screen
        self.label.update()
                       
    @Slot()    
    def project_home_screen(self):    

        #Display home image
        self.label.setPixmap(self.home_image)
        
        #Show
        self.label.update()
    
    @Slot()    
    def project_target_screen(self):
        
        #Display home image
        self.label.setPixmap(self.target_image)
        
        #Show
        self.label.update() 

    #Main projection sequence   
    @Slot(int)
    def project_N_images(self, N):
                
        self.projector_cond.release(1)
        
        print("Projector:      Starting projection of {} image(s)".format(N))
        
        for i in range(0,N):
           
            #Wait for signal
            self.projector_cond.acquire(1)
 
            #Acquire frame
            projection_image = self.projector_queue.get()
 
            #Project
            self.project(projection_image)
 
            #Process events
            QApplication.processEvents()
  
            print("Projector:      Projected image {:02d}".format(i))
                        
            #Signal camera
            self.camera_cond.release(1)
            
        print("Projector:      COMPLETE")   
        self.finish_cond.release(1)
        
    