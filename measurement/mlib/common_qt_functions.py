from PySide2.QtGui import (
    QImage,
    QPixmap
    )

import numpy as np

#Change numpy array to QPixmap
def array_to_QPixmap(array, image_type = 'Grayscale8'):
    
    #Inputs:
    #array - numpy array REQUIRED: image coordinate (transpose of normal), 
    #                              contiguous order
    
    #Tranpose the array
    if array.ndim == 2:
        array = array.transpose()
    elif array.ndim == 3:
        array = np.transpose(array, axes=(1,0,2))
    
    #Make contiguous if not
    if not array.flags['C_CONTIGUOUS']:
        array = array.copy(order='C')
        
    height = array.shape[0]
    width = array.shape[1]
    
    #%%

    if image_type == 'Grayscale8':
        bytes_per_line = 1 * width
        
        try:
            qimage = QImage(array, width, height, bytes_per_line,                                                                                                                                               
                     QImage.Format_Grayscale8)
        except Exception as e:
            print(e)
            
    #%%
    elif image_type == 'Grayscale16':
        bytes_per_line = 1 * width
        
        try:
            qimage = QImage(array, width, height, bytes_per_line,                                                                                                                                               
                     QImage.Format_Grayscale16)
        except Exception as e:
            print(e)
    
    #%%
    elif image_type == 'RGB888':
        bytes_per_line = 3 * width
        
        try:
            qimage = QImage(array, width, height, bytes_per_line,
                            QImage.Format_RGB888)
            
        except Exception as e:
            print(e)
            
    #%%
    elif image_type == 'ARGB32':
        bytes_per_line = 4 * width
        
        try:
            qimage = QImage(array, width, height, bytes_per_line,
                            QImage.Format_ARGB32)
            
        except Exception as e:
            print(e)
    
    else:
        print("INCOMPATIBLE CODING FOR QIMAGE")
    #%%

    pixmap = QPixmap.fromImage(qimage)
           
    return pixmap