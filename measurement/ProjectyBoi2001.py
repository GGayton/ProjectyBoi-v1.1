import os
import queue

from PySide2.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QBoxLayout,
    QSpinBox,
    QGridLayout,
    QGroupBox,
    QWidget,
    QComboBox,
    )

from PySide2.QtCore import (
    QSemaphore,
    QThread,
    Signal,
    Slot,
    )

from commonlib.projector import ProjectorControl
from commonlib.streamer import ViewWindow
from commonlib.camera import CameraControl
from commonlib.saver import ImageSaver
from commonlib.measurement import TaskManager
#%% Control window
class ControlWindow(QMainWindow):
    
    """
    Provides the GUI for the user to select certain parameters (what images to use etc)
    
    All the signals/slot connections are conected outside the class
    
    In Qt, the main GUI has to exist in the main thread. Im not entirely sure what that means but this seems to work.
    
    There are a lot of deprecated functions/signals and slots etc. I haven't cleaned them up, so if you think they do nothing chances are they do nothing.
    """
    
    update_images_signal = Signal(str)
    project_nth_image_signal = Signal(int)
    
    def __init__(self):
        super().__init__()
        
        #Current directory
        self.current_directory = os.path.dirname(os.path.realpath(__file__))
        
        #Create input box
        window = QWidget()
        
        grid = QGridLayout()
        grid.addWidget(self.create_measurement_box(),0,0)
        grid.addWidget(self.create_preparation_box(),0,1)
        grid.addWidget(self.create_repeat_box(),1,0)
        grid.addWidget(self.create_warmup_box(),1,1)
        grid.addWidget(self.create_close(),2,1)
        
        window.setLayout(grid)
        
        self.setCentralWidget(window)
        
        self.setWindowTitle("ProjectyBoy2000")
        
        self.show()
        
    def create_measurement_box(self):
        
        measurement_box = QGroupBox("Measurement")
        
        #Create initiate measurement button
        init_measurement = QPushButton("Start")
        init_measurement.setObjectName("init_measurement")
               
        #Find the regime types
        regime_file_dir = self.current_directory + '\\Projection Regimes\\Regimes'
        regime_list = os.listdir(regime_file_dir)
        
        #Create regime control
        regime_control = QComboBox(self)
        regime_control.setObjectName("regime_control")
        for regime in regime_list:

            if regime[-5:] == '.hdf5':
                regime_control.addItem(regime)
            
        #Repeat num
        text1 = QLabel("Set repeats:")
        repeat = QSpinBox()
        repeat.setRange(1,500)
        repeat.setSingleStep(1)
        repeat.setValue(1)
        repeat.setObjectName("repeat_camera_images")
                    
        vbox = QBoxLayout(QBoxLayout.TopToBottom)

        vbox.addWidget(init_measurement)
        vbox.addWidget(regime_control)
        vbox.addWidget(text1)
        vbox.addWidget(repeat)
        vbox.addStretch(1)
        
        measurement_box.setLayout(vbox)
        
        return measurement_box
    
    def create_warmup_box(self):
        
        warm_up_box = QGroupBox("Warm-up")

        
        warmup = QPushButton("Initiate")
        warmup.setObjectName("init_warm_up")

        
        text1 = QLabel("Set repeats:")
        warmup_repeat = QSpinBox()
        warmup_repeat.setRange(1,500)
        warmup_repeat.setSingleStep(1)
        warmup_repeat.setValue(1)
        warmup_repeat.setObjectName("warm_up_repeats")
        
        vbox = QBoxLayout(QBoxLayout.TopToBottom)

        vbox.addWidget(text1)
        vbox.addWidget(warmup_repeat)
        vbox.addWidget(warmup)
        vbox.addStretch(1)
        
        warm_up_box.setLayout(vbox)
        
        return warm_up_box
            
    def create_preparation_box(self):
        
        preparation_box = QGroupBox("Preparation")
            
        target_screen = QPushButton("Target")
        target_screen.setObjectName("target_screen")
        
        #Create initiate measurement button
        init_streaming = QPushButton("Start streaming")
        init_streaming.setObjectName("init_streaming")
            
        #Project Nth image button
        project_nth = QPushButton("Project nth image")
        project_nth.setObjectName("project_nth_image")
        
        text1 = QLabel("Set n:")
        nth_image = QSpinBox()
        nth_image.setRange(0,500)
        nth_image.setSingleStep(1)
        nth_image.setValue(0)
        nth_image.setObjectName("set_nth_image")
                
        vbox = QBoxLayout(QBoxLayout.TopToBottom)

        vbox.addWidget(target_screen)
        vbox.addWidget(init_streaming)
        vbox.addWidget(project_nth)
        vbox.addWidget(text1)
        vbox.addWidget(nth_image)
        vbox.addStretch(1)
        
        preparation_box.setLayout(vbox)
        
        return preparation_box

    def create_repeat_box(self):
        camera_box = QGroupBox("Repeat measurements")
        camera_box.setObjectName("repeat")

        start = QPushButton("Take N measurements")
        start.setObjectName("init_repeats")
        
        text1 = QLabel("Set N:")
        repeat = QSpinBox()
        repeat.setRange(1,100)
        repeat.setSingleStep(1)
        repeat.setValue(1)
        repeat.setObjectName("repeat_num")
        
        vbox = QBoxLayout(QBoxLayout.TopToBottom)
        
        vbox.addWidget(start)
        vbox.addWidget(text1)
        vbox.addWidget(repeat)
        vbox.addStretch(1)
        
        camera_box.setLayout(vbox)
        
        return camera_box
    
    def create_close(self):
        
        #Create exit button
        close_now = QPushButton("Close")
        close_now.setObjectName("close_now")
        
        return close_now
    
    @Slot(str)
    def define_regime_directory(self, regime_directory):
        
        #List all images
        directory = self.current_directory + '\\Projection Regimes\\Regimes\\' + regime_directory
        
        print("Loading from:      [",directory,"]")
               
        #Remember number of images in stack
        self.update_images_signal.emit(directory)       
        
#%%
            
def close_all():
    
    if viewing_window.isVisible():
        viewing_window.close_stream()
    camera_control_thread.quit()
    camera_control_thread.wait()
    
    app.closeAllWindows()
    
            
if __name__ == "__main__":
     
    #Start application
    app = QApplication.instance()
    if app == None:
        app = QApplication([])
        
    #Create queues
    camera_queue = queue.Queue()
    projector_queue = queue.Queue()
    camera_cond = QSemaphore()
    projector_cond = QSemaphore()
    finish_cond = QSemaphore()
        
    #Initiate objects
    viewing_window = ViewWindow(
        camera_queue,
        camera_cond
        )
    control_window = ControlWindow()
    projector_control = ProjectorControl(
        projector_queue, 
        projector_cond,
        camera_cond,
        finish_cond
        )
        
    camera_control = CameraControl(
        camera_queue, 
        camera_cond, 
        projector_cond,
        finish_cond
        )
    
    image_saver = ImageSaver(
        camera_queue, 
        finish_cond,
        os.path.dirname(os.path.realpath(__file__)) + "\\Measurements\\"
        )
    
    task_manager= TaskManager(
        finish_cond, 
        projector_queue, 
        projector_cond, 
        camera_cond, 
        )
    
    #Initiate threads
    image_saver_thread = QThread()
    task_manager_thread = QThread()
    camera_control_thread = QThread()
    
    image_saver_thread.start()
    task_manager_thread.start()
    camera_control_thread.start()
    
    #Move to threads    
    image_saver.moveToThread(image_saver_thread)
    task_manager.moveToThread(task_manager_thread)
    camera_control.moveToThread(camera_control_thread)
    
    #Connect signals

    #Close
    control_window.centralWidget().findChild(QPushButton, "close_now").clicked.connect(close_all)
    
    #Measuring
    control_window.centralWidget().findChild(QPushButton, "init_measurement").\
        clicked.connect(task_manager.measurement)
        
    #Repetition
    control_window.centralWidget().findChild(QSpinBox, "repeat_num").\
        valueChanged[int].connect(task_manager.set_repetitions)
    control_window.centralWidget().findChild(QPushButton, "init_repeats").\
        clicked.connect(task_manager.repeat_measurement)
        
        
    control_window.centralWidget().findChild(QComboBox, "regime_control").\
        activated[str].connect(control_window.define_regime_directory)
    control_window.update_images_signal.\
        connect(task_manager.load_in_images)
        
    task_manager.project_N_images_signal.\
        connect(projector_control.project_N_images)
    task_manager.acquire_N_images_signal.\
        connect(camera_control.take_N_images)
    task_manager.save_N_images_signal.\
        connect(image_saver.save_N_images)
    task_manager.bin_N_images_signal.\
        connect(image_saver.bin_N_images)
    
    #Preparation
    control_window.centralWidget().findChild(QPushButton, "target_screen").\
        clicked.connect(projector_control.project_target_screen)
    control_window.centralWidget().findChild(QPushButton, "init_streaming").\
        clicked.connect(viewing_window.toggle_stream_window)
    
    control_window.centralWidget().findChild(QPushButton, "project_nth_image").\
        clicked.connect(task_manager.project_nth_image)
    control_window.centralWidget().findChild(QSpinBox, "set_nth_image").\
        valueChanged[int].connect(task_manager.set_nth_image)

    viewing_window.start_streaming_signal.\
        connect(camera_control.start_streaming)
    viewing_window.stop_streaming_signal.\
        connect(camera_control.stop_streaming)
    viewing_window.clear_camera_queue_signal.\
        connect(camera_control.clear_camera_queue)
    viewing_window.clear_camera_cond_signal.\
        connect(camera_control.clear_camera_cond)
    
    
    control_window.centralWidget().findChild(QSpinBox, "repeat_camera_images").\
        valueChanged[int].connect(camera_control.set_repeat_images)
    control_window.centralWidget().findChild(QSpinBox, "repeat_camera_images").\
        valueChanged[int].connect(image_saver.set_repeat_images)  
        
    #Warm up
    control_window.centralWidget().findChild(QSpinBox, "warm_up_repeats").\
        valueChanged[int].connect(task_manager.set_warm_up)
    control_window.centralWidget().findChild(QPushButton, "init_warm_up").\
        clicked.connect(task_manager.warm_up)
        

    app.exec_()
    
    print("Goodbye")
    image_saver_thread.quit()
    task_manager_thread.quit()
    camera_control_thread.quit()
    
    image_saver_thread.wait()
    task_manager_thread.wait()
    camera_control_thread.wait()
    
    app.closeAllWindows()
    
