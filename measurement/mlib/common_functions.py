import numpy as np
from PIL import Image
import os
import matplotlib
import io
import cv2
import h5py
import time

#%% Changing different number formats

def rescale_to_uint8(array, max_value=[]):
    
    if not max_value:
        max_value = np.iinfo(array.dtype).max
    
    array = array.astype(np.float)
    array = array*255/max_value
    array = array.astype(np.uint8)
    
    return array

def convert_to_unsigned_integer(array, bit_number):
    
    max_value = 2*bit_number
    
    array[array>max_value] = max_value
    array[array<0] = 0
    
    if bit_number>0 & bit_number <= 8:
        return array.astype(np.uint8)

    elif bit_number>8 & bit_number <= 16:
        return array.astype(np.uint16)
    
    elif bit_number>16 & bit_number <= 32:
        return array.astype(np.uint32)
    
    else:
        print("Bit number too high or too low")

def convert_float_to_uint8_image(array, min_value=0, max_value=255):
    
    array = np.round(array*(max_value-min_value) + min_value)
    
    return array.astype(np.uint8)
 
def create_R(T, order = 'XYZ'):
    
    if order == 'XYZ':
        Tx = T[0]
        Ty = T[1]
        Tz = T[2]
        
        
        Rx = np.array([
            [1,0,0],
            [0,np.cos(Tx),-np.sin(Tx)],
            [0, np.sin(Tx), np.cos(Tx)]
            ])
        
        Ry = np.array([
            [np.cos(Ty),0,np.sin(Ty)],
            [0,1,0],
            [-np.sin(Ty),0,np.cos(Ty)]
            ])
        
        Rz = np.array([
            [np.cos(Tz), -np.sin(Tz),0],
            [np.sin(Tz),np.cos(Tz),0],
            [0,0,1]
            ])
        
        R = Rx@Ry@Rz
    
    else:
        print("UNWRITTEN")
    
    return R

def home_directory():
  
    current_directory = os.getcwd()
    
    cutoff = current_directory.find("ProjectyBoy2000")
    
    if cutoff == -1:
        print("HOME DIRECTORY NOT FOUND")
        
        return -1
    
    else:
    
        home_directory = current_directory[:cutoff+16]
    
        return home_directory

def load_arrays(measurement_directory, start, end, preffix="", suffix=""):
    
    stack = []
    for i in range(start,end):
        
        string = measurement_directory + preffix+ "{:02d}.npy".format(i) + suffix
                
        image = np.load(string)
        
        stack.append(image)
    
    if len(stack) == 1:
        stack = stack[0]
    
    return stack

def subsample(array, factor, mode='uniform'):
    
    if mode=='uniform':
        
        if array.ndim == 1:
            sub_sampled_array = array[::factor]
            
        elif array.ndim == 2:
            sub_sampled_array = array[::factor,::factor]
            
        elif array.ndim == 2:
            sub_sampled_array = array[::factor,::factor,::factor]
       
        else:
            print("INCORRECT ARRAY DIMENSION")
            
    elif mode=='random':
        
        print("UNWRITTEN")
        
    else:
        
        print("CHOOSE UNIFORM OR RANDOM")
        
    return sub_sampled_array
    
def distort(vector, D):
    
    k1 = D[0]
    k2 = D[1]
    p1 = D[2]
    p2 = D[3]
    k3 = D[4]
    
    r2 = vector[:,0]**2 + vector[:,1]**2
    xy = vector[:,0]*vector[:,1]
    
    radial_correction = (1 + k1*r2 + k2*r2**2 + k3*r2**3)
    x_tangential_correction = 2*p1*xy + p2*(r2+2*vector[:,0]**2)
    y_tangential_correction = 2*p2*xy + p1*(r2+2*vector[:,1]**2)
    
    distorted_vector = vector*radial_correction +  np.array([x_tangential_correction, y_tangential_correction])
    
    return distorted_vector

# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def FT(array, mode):
    
    if mode=='fft2':
        
        array_FT = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(array)))
        
    return array_FT

def sample_ellipse(x0,y0,a,b,theta):
    
    #Solve for X
    A = 4*( np.sin(theta)**2 * np.cos(theta)**2 * ((1/b**2 - 1/a**2)**2 - 2/(a**2*b**2)) - \
        ( (np.cos(theta)/b)**4 + (np.sin(theta)/a)**4 ))
    
    C = 4*( (np.cos(theta)/b)**2 + (np.sin(theta)/a)**2 )
    
    numerator = -4*A*C
        
    if numerator < 0:
        print("no roots")
    elif numerator == 0:
        root = ( numerator**0.5 ) / (2*A)
        print("1 root")
    else:
        root = [0,0]
        
        #Find root
        root[0] = ( numerator**0.5 ) / (2*A)
        root[1] = -( numerator**0.5 ) / (2*A)
        
        #convert ot integer (which also rounds towards 0)
        root[0] = int(root[0])
        root[1] = int(root[1])
        
    #order the roots
    xlim = (np.min(root), np.max(root))
    
    num_of_x_points = xlim[1]-xlim[0] + 1
    
    #Create x sample space        
    x = np.linspace(xlim[0], xlim[1], num_of_x_points, dtype = np.int16)
    
    #Solve for y
    A = (np.sin(theta)/a)**2 + (np.cos(theta)/b)**2
    B = 2*x*np.sin(theta)*np.cos(theta)*(1/b**2 - 1/a**2)
    C = x**2 * ( (np.sin(theta)/a)**2 + (np.cos(theta)/b)**2 ) -1
       
    y1 = ( -B + (B**2 - 4*A*C)**0.5 ) / (2*A)
    y2 = ( -B - (B**2 - 4*A*C)**0.5 ) / (2*A)
    
    #Concatenate
    x = np.concatenate([x,x]).reshape(num_of_x_points*2,1) + x0
    y = np.concatenate([y1,y2]).reshape(num_of_x_points*2,1) + y0
    vec = np.concatenate([x,y],axis=1)
    
    vec = vec.astype(np.int16)
    
    return vec

def find(array, x,y):
    l=2
    
    Xindex = np.logical_and(array[:,0]>x-l, array[:,0]<y+l)
    Yindex = np.logical_and(array[:,1]>y-l, array[:,1]<y+l)
    
    index = np.logical_and(Xindex, Yindex)
    
    possible_values = np.nonzero(index)
    
    return possible_values

def extract_theta_from_R(R, order = 'XYZ'):
    
    if order == 'XYZ':
        y = np.arcsin(R[0,2])
        
        z = -np.arcsin(R[0,1] / np.cos(y))
        
        x = -np.arcsin(R[1,2] / np.cos(y))
        
        return x,y,z
    
    if order == 'YZX':
                
        z = np.arcsin(R[1,0])
                              
        y = -np.arcsin(R[2,0] / np.cos(z))
        
        x = np.arccos(R[1,1] / np.cos(z))
        
        return x,y,z

#%% Smaplnig

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
#%% Plotting
def plot_array(array, fig_num = [], close = True):
    
    if close:
        matplotlib.pyplot.close(fig_num)
    
    if fig_num:
        matplotlib.pyplot.figure(fig_num)
        
    matplotlib.pyplot.imshow(array)
    matplotlib.pyplot.show()

def plot_scatter(X, Y, indexing = 'ij', fig_num = [], close = True):
    
    if close:
        matplotlib.pyplot.close(fig_num)
    
    if fig_num:
        matplotlib.pyplot.figure(fig_num)
    
    if indexing == 'ij':
        matplotlib.pyplot.scatter(X,Y,s=1, c='red')
    if indexing == 'xy':
        matplotlib.pyplot.scatter(Y,X,s=1, c='red')
        
    matplotlib.pyplot.show()

def bivariate_normal_distribution(x,y,meanX,meanY,stdX,stdY,corr):
    
        dist = (1/( stdX*stdY*2*np.pi*(1-corr**2)**0.5)) *\
        np.exp(- 1/(2*(1-corr**2)) * \
               ( \
               +((x - meanX)/stdX)**2 \
               -((x - meanX)/stdX)*((y - meanY)/stdY)*2*corr \
               +((y - meanY)/stdY)**2 \
               ))
            
        return dist
    

def plot_bivariate_normal_distribution(data, limits=(-1,1), N=1000):
    
    #Find mean
    meanX = np.mean(data[:,0])
    meanY = np.mean(data[:,1])
    
    #Find covariance matrix
    cov = np.cov(data[:,0], data[:,1])

    stdX = cov[0,0]**0.5
    stdY = cov[1,1]**0.5
        
    corr = cov[1,0]/(stdX*stdY)
    #Plot
    xmesh, ymesh = np.meshgrid(
        np.linspace(limits[0], limits[1], N),
        np.linspace(limits[0], limits[1], N))
    
    dist = bivariate_normal_distribution(xmesh,ymesh,meanX,meanY,stdX,stdY,corr)
    
    c1 = bivariate_normal_distribution(1*stdX,0,meanX,meanY,stdX,stdY,corr)
    c2 = bivariate_normal_distribution(2*stdX,0,meanX,meanY,stdX,stdY,corr)
    c3 = bivariate_normal_distribution(3*stdX,0,meanX,meanY,stdX,stdY,corr)
    
    contours = (c3, c2, c1)
        
    matplotlib.pyplot.contour(xmesh,ymesh,dist,contours, colors = 'r')
        
    matplotlib.pyplot.text(stdX*1, 0, '$\sigma_1$', color='black', 
        backgroundcolor='white', fontsize = 14)
    matplotlib.pyplot.text(stdX*2, 0, '$\sigma_2$', color='black', 
        backgroundcolor='white', fontsize = 14)
    matplotlib.pyplot.text(stdX*3, 0, '$\sigma_3$', color='black', 
        backgroundcolor='white', fontsize = 14)
    
def plot_normal_distribution(data, limits=(-1,1),N = 1000):
    
    mean = np.mean(data)

    std = np.std(data)
    
    x = np.linspace(limits[0], limits[1], N)
    
    normal = (1/(std * (2*np.pi)**0.5)) * np.exp(-0.5 * ((x - mean)/(std))**2)

    matplotlib.pyplot.plot(x,normal)
    
def plot_random_distribution(data, limits=(-1,1), N=1000):
    
    mean = np.mean(data)

    std = np.std(data)
    
    a = (std**2 * 3)**0.5
    boundary = (mean-a, mean+a)
    
    x = np.linspace(limits[0], limits[1], N)
    
    output = np.logical_and(x>=boundary[0],x<=boundary[1])
    output = output.astype(np.float)/(boundary[1]-boundary[0])
    
    matplotlib.pyplot.plot(x,output)

def plot_bivariate_random_distribution(data, limits=(-1,1), N=1000):
    
    #Find mean
    meanX = np.mean(data[:,0])
    meanY = np.mean(data[:,1])
        
    stdX = np.std(data[:,0])
    stdY = np.std(data[:,1])
    
    a = (stdX**2 * 3)**0.5
    b = (stdY**2 * 3)**0.5
    
    boundaryX = (meanX-a, meanX+a)
    boundaryY = (meanY-b, meanY+b)

    matplotlib.pyplot.plot(
        [boundaryX[1], boundaryX[0], boundaryX[0], boundaryX[1], boundaryX[1]],
        [boundaryY[1], boundaryY[1], boundaryY[0], boundaryY[0], boundaryY[1]])    

def show_array(array):
    
    array = array.astype(np.float)
    
    array_min = np.min(array)
    array_max = np.max(array)
    array_range = array_max - array_min
    
    #Scale
    array = (array - array_min)/array_range * 255
    
    array = array.astype("uint8")
            
    img = Image.fromarray(array, 'L')
    img.show()

def plot_line(array, axis="X", index=0):
    
    if axis == "X":
        line = array[index,:]
    
    elif axis == "Y":
        line = array[:,index]
        
    matplotlib.pyplot.plot(np.linspace(0,line.shape[0]-1, line.shape[0]), line)
    
def plot_lines_between_datasets(X, Y, ax=[]):
    
    if not ax:
        
        fig = matplotlib.pyplot.figure(10)
        ax = fig.add_subplot(111, projection='3d')
    
    for i in range(X.shape[1]):
        
        ax.plot(
            [X[0,i], Y[0,i]],
            [X[1,i], Y[1,i]],
            [X[2,i], Y[2,i]],
            'r')
        
        

    






