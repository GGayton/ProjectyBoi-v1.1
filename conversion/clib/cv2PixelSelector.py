import cv2
import matplotlib
import numpy as np

def rescale_to_uint8(array, max_value=[]):
    
    if not max_value:
        max_value = np.iinfo(array.dtype).max
    
    array = array.astype(np.float)
    array = array*255/max_value
    array = array.astype(np.uint8)
    
    return array



def defineArea(image, downScale=6):
    # function to display the coordinates of 
    # of the points clicked on the image  
    coords = []
    
    res = image.shape
    
    u,v = np.meshgrid(np.linspace(1,res[0],res[0]), np.linspace(1,res[1],res[1]), indexing = 'ij')
    vec = np.concatenate((u.astype(np.uint16).reshape(-1,1), v.astype(np.uint16).reshape(-1,1)), axis=1)
    
    def click_event(event, x, y, flags, params):
        print("ass")
        global coords, downScale
        # checking for left mouse clicks 
        if event == cv2.EVENT_LBUTTONDOWN: 
      
            # displaying the coordinates 
            # on the Shell 
            print("X: {}, Y: {}".format(downScale*y,downScale*x))
            coords.append((downScale*y,downScale*x))
    
    test_image = rescale_to_uint8(image, max_value=image.max())
    
    width = int(test_image.shape[0]/downScale)
    height = int(test_image.shape[1]/downScale)
    test_image = cv2.resize(test_image, (width, height))
    
    cv2.imshow('Crop image', test_image)
    # setting mouse hadler for the image 
    # and calling the click_event() function 
    cv2.setMouseCallback('Gamma test', click_event) 
    cv2.waitKey(0)  
          
    #closing all open windows  
    cv2.destroyAllWindows()  

    p = matplotlib.path.Path(coords)
    index = p.contains_points(vec)
    
    return index.reshape(res[0], res[1])