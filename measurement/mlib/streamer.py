from PySide2.QtWidgets import (
    QLabel,
    QPushButton,
    QGridLayout,
    QWidget,
    QApplication
    )
from PySide2.QtGui import (
    QPainter,
    QPen,
    QPixmap)
from PySide2.QtCore import (
    Qt,
    Signal,
    Slot,
    )
import numpy as np
from mlib.common_functions import (
    rescale_to_uint8,
    get_img_from_fig
    )
from mlib.common_qt_functions import (
    array_to_QPixmap
    )
import cv2
import matplotlib.pyplot



#%%

class ViewWindow(QWidget):
    
    """
    This class evaluates the image taken during "streaming" mode, shows it to 
    the user on the GUI and gives info on saturated pixels and contrast.
    
    The inputs are the 
    
    Queue object : queue (which takes images (given by the camera))
    QSemaphore object: camera_cond (Tells the camera to take an image)
        
    """
    
    start_streaming_signal = Signal()
    stop_streaming_signal = Signal()
    clear_camera_queue_signal = Signal()
    clear_camera_cond_signal = Signal()
    
    def __init__(self, queue, camera_cond):
        super().__init__()
        
        self.queue = queue
        self.camera_cond = camera_cond
                
        self.label = QLabel('No Camera Feed')
        self.histogram = QLabel('')
        self.close_window = QPushButton('Close stream')
        self.take_picture = QPushButton('Capture')
        self.find_grid = QPushButton('Find grid')
        
        #Window properties
        self.setWindowTitle("Camera Stream")
        self.setGeometry(100,100,500,500)
        
        #Create UI
        layout = QGridLayout()
        layout.addWidget(self.take_picture, 0,0,1,1)
        layout.addWidget(self.find_grid, 0,1,1,1)
        layout.addWidget(self.label, 1,0,1,1)
        layout.addWidget(self.histogram, 1,1,1,1)
        layout.addWidget(self.close_window, 2,0,1,2)

        self.setLayout(layout)
        
        self.take_picture.clicked.connect(self.take_image)
        self.close_window.clicked.connect(self.toggle_stream_window)
        self.find_grid.clicked.connect(self.identify_board)
        
        self.board_size = (8,23)
        self.frame = np.empty((1,1))

    @Slot()
    def toggle_stream_window(self):
        if self.isVisible():
            self.close_stream()
        else:
            self.open_stream()

    @Slot()
    def open_stream(self):
        self.show()
        self.start_streaming_signal.emit()
        
    @Slot()
    def close_stream(self):

        self.queue.put([])
        self.stop_streaming_signal.emit()
        self.clear_camera_queue_signal.emit()
        self.clear_camera_cond_signal.emit()
        self.hide()
        
    def identify_board(self):
        
        ret = False
        
        try:
            ret, points = cv2.findCirclesGrid(self.frame, self.board_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        except Exception as e:
            print("Streamer:      ", e)

            
        if ret:            
            
            #Obtain pixmap
            pixmap = self.label.pixmap()
            
            #Initiate QPainter
            painter = QPainter(pixmap)
            
            #Set the pen
            painter.setPen(QPen(Qt.green,  4))
            
            #Draw every point
            for i in range(points.shape[0]):
                painter.drawPoint(points[i,0,0], points[i,0,1])
            
            #Update the pixmap
            self.label.setPixmap(pixmap)
            
            #End the painter
            painter.end()
            
            self.label.repaint()
                    
        else:
            print("Streamer:       Board not found")
                                      
    def take_image(self):
        self.camera_cond.release(1)
        
        #Obtain the frame from the queue
        try:
            frame = self.queue.get(timeout=2000)
            
            data =self.create_histogram(frame)
                                                    
            #Convert frame
            frame = self.convert_stream(frame)
            self.frame = frame
            
            data = np.transpose(data, (1,0,2))
            frame = np.transpose(frame, (1,0,2))
            qpixmap_label = array_to_QPixmap(frame, image_type='RGB888')           
            qpixmap_histogram = array_to_QPixmap(data, image_type='RGB888')
                        
            #Set the pixmap
            self.label.setPixmap(qpixmap_label)
            self.histogram.setPixmap(qpixmap_histogram)
            
            #Update
            self.label.repaint()
            
            QApplication.processEvents()
            
        except Exception as e:
            print("Streamer:      ",e)
            print("Streamer:       Stream ended")
            self.toggle_stream_window()
        
    def convert_stream(self, frame):
        
        #Rescale
        frame = frame[::6, ::6]
        
        #Find saturated pixels        
        index = frame==1023
        
        #Convert to 8-bit format for viewing
        frame = rescale_to_uint8(frame, max_value = 1023)
        frame = frame.reshape(frame.shape[0], frame.shape[1], 1)
        
        #Convert to RGB
        rgb = np.repeat(frame, 3, axis = 2)        

        #Change saturated pixels to red
        rgb[index] = np.array([1023,0,0]).reshape(1,1,3)
        
        return rgb
    
    def create_histogram(self, array):
        
        # Make a random plot...
        fig = matplotlib.pyplot.figure(1234)
        
        array = array.reshape(array.shape[0]*array.shape[1], 1)
        
        bins = np.linspace(1,1023,1023)
            
        matplotlib.pyplot.hist(array, bins=bins)
        
        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        fig.canvas.draw()
        
        # Now we can save it to a numpy array.
        # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        data = get_img_from_fig(fig)
        
        matplotlib.pyplot.close(1234)
        
        return data