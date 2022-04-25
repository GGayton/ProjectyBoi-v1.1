import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import matplotlib

def defineArea(image):
    
    coords = []
    index = None
    stopping = True
    
    res = image.shape
    
    u,v = np.meshgrid(np.linspace(1,res[0],res[0]), np.linspace(1,res[1],res[1]), indexing = 'ij')
    vec = np.concatenate((u.astype(np.uint16).reshape(-1,1), v.astype(np.uint16).reshape(-1,1)), axis=1)
    
    def onclick(event):
        
        nonlocal coords
        
        fig = plt.gcf()
                        
        if event.button is MouseButton.RIGHT:
            coords = coords[:-1]

        elif event.button is MouseButton.LEFT:
            coords.append((event.ydata, event.xdata))
            
        a = np.array(coords)

        plt.plot(a[:,1],a[:,0],'r')
        plt.plot(a[:,1],a[:,0],'r.')
        
        fig.canvas.draw()
     
    def onclose(event):

        nonlocal coords,index,stopping
                                            
        p = matplotlib.path.Path(coords)
        index = p.contains_points(vec)
        
        fig = plt.gcf()

        fig.canvas.stop_event_loop()
        plt.close(fig)
        stopping = False
    
    fig = plt.figure()
    
    plt.imshow(image)
    
    plt.title('Please enclose region of interest')
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('close_event', onclose)
    plt.show()
    
    while stopping:
        
        plt.pause(0.5)
        
    return index.reshape(res[0],res[1])
    
def defineCentres(image, downScale = 1):
        
    coords = []
    stopping = True
    
    downScaledImage = image[::downScale, ::downScale]
    
    def onclick(event):
        
        nonlocal coords, scatt
        
        fig = plt.gcf()
                                
        if event.button is MouseButton.RIGHT:
            coords = coords[:-1]

        elif event.button is MouseButton.LEFT:
            coords.append((event.ydata * downScale, event.xdata * downScale))
            
        a = np.array(coords)
        
        plot(a)
        
        fig.canvas.draw()
  
    def onclose(event):

        nonlocal stopping,fig
                    
        fig.canvas.stop_event_loop()
        plt.close(fig)
        stopping = False
    
    def plot(a):
        
        nonlocal scatt
        
        scatt.remove()
        scatt = plt.scatter(a[:,1] / downScale,a[:,0] / downScale, s=5, c='r')
                
    fig = plt.figure()
    plt.imshow(downScaledImage)
    scatt = plt.scatter([],[], s=1, c='r')
    
    plt.title('Please identify centres of features')
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('close_event', onclose)
    plt.show()
    
    while stopping:
        
        plt.pause(0.5)
        
    return coords

def lineSegmentIndex(arraySize,start,end,thickness=1):
        
    x1 = int(start[0])
    x2 = int(end[0])
    
    y1 = int(start[1])
    y2 = int(end[1])
    
    #best fit line
    grad = (y2-y1)/(x2-x1)
    origin = -x1*grad+y1
    
    x = (np.linspace(x1, x2, np.abs(x2-x1) + 1)).astype(int)
    y = (origin + x*grad).astype(int)
        
    index = np.zeros(arraySize, dtype = bool)
    index[x,y] = True
    
    for i in range(1,thickness//2+1):
        
        index[x,y+i] = True
        index[x,y-i] = True
    
    return index

def defineLine(image, thickness=1, downScale=1):
    
    coords = []
    stopping = True
    
    downScaledImage = image[::downScale, ::downScale]
    
    def onclick(event):
        
        nonlocal coords
        
        fig = plt.gcf()
                                
        if event.button is MouseButton.RIGHT:
            coords = coords[:-1]

        elif event.button is MouseButton.LEFT:
            coords.append((event.ydata * downScale, event.xdata * downScale))
            
        a = np.array(coords)
        
        plot(a)
        
        fig.canvas.draw()
  
    def onclose(event):

        nonlocal stopping,fig
                    
        fig.canvas.stop_event_loop()
        plt.close(fig)
        stopping = False
        
    def plot(a):
        
        nonlocal scatt, ploot
        
        scatt.remove()
        ploot.remove()
        scatt = plt.scatter(a[:,1] / downScale,a[:,0] / downScale, s=5, c='r')
        ploot = plt.plot(a[:,1] / downScale,a[:,0] / downScale, 'r')[0]
        
    fig = plt.figure()
    plt.imshow(downScaledImage)
    scatt = plt.scatter([],[], s=1, c='r')
    ploot = plt.plot([],[],'r')[0]
    
    plt.title('Please select a line')
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('close_event', onclose)
    plt.show()
    
    while stopping:
        
        plt.pause(0.5)
        
    return lineSegmentIndex(image.shape, coords[0], coords[1], thickness = thickness)
    





