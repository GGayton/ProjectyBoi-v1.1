from clib.InputMeasurement import InputMeasurement
from clib.OpenCVNonlinearCorrection import NonlinearCorrection
from clib.Common import Common
import h5py

#%%
class Measurement(Common):
    """
    This class handles the maths for converting the correspondence map 
    (the image whose values correspond to projector coordinates) to 3D points.
    
    This class I tried to make universal between CPU and GPU computing which is 
    defined by the useCUDA attribute. Have only tested on GPU so far.
    
    """
    def __init__(self,axis,useCUDA=False):

        self.useCUDA=useCUDA
        if useCUDA: import cupy as xp
        else: import numpy as xp
        self.xp=xp
        
        self.axis = axis
        if axis == "X": self.A = xp.array([[0,0,-1], [0,0,0], [1,0,0]])
        elif axis == "Y": self.A = xp.array([[0,0,0], [0,0,1], [0,-1,0]])
        
        self.inputP = InputParameter(useCUDA = useCUDA)
        self.inputM = InputMeasurement(useCUDA = useCUDA)
        self.inputC = NonlinearCorrection(useCUDA = useCUDA)
        self.undistort = self.inputC.undistort
        self.distort = self.inputC.distort
        
    #%% Core        
    def __measure(self,Kc,iKc,Kp,R,t,A,pixelVector,measurementVector):
        xp=self.xp

        output = iKc @ pixelVector
        partA = R.T @ Kp.T @ A @ measurementVector
        L = ((-t.T @ R) @ partA) / (xp.sum(output * partA, axis=0,keepdims=True))
        
        output = output*L
                        
        return output
                
    def getPointCloud(self):
        
        xp = self.xp
        
        #Obtain values
        Kc,Dc,Kp,t,R,Dp = self.inputP.get()
        A = self.A
        
        pixelVector, measurementVector = self.inputM.get()
        
        iKc = xp.linalg.inv(Kc)
        
        measurementList = []
        
        for i in range(len(pixelVector)):
    
            pixelVectorTemp = self.undistort(pixelVector[i],Kc,Dc[0],Dc[1],Dc[2],Dc[3],Dc[4])
            measurementVectorTemp = self.undistort(measurementVector[i],Kp,Dp[0],Dp[1],Dp[2],Dp[3],Dp[4])

            measurementList.append(self.__measure(Kc,iKc,Kp,R,t,A,pixelVectorTemp,measurementVectorTemp).T)
               
        return measurementList
            
    def getVirtualPointCloud(self):
        
        xp = self.xp
        
        Kc,Dc,Kp,t,R,Dp = self.inputP.getVirtual()
        
        A = self.A
        pixelVector, measurementVector = self.inputM.getVirtual()
                
        iKc = xp.linalg.inv(Kc)
        
        measurementList = []
        
        
        for i in range(len(pixelVector)):
    
            pixelVectorTemp = self.undistort(pixelVector[i],Kc,Dc[0],Dc[1],Dc[2],Dc[3],Dc[4])
            measurementVectorTemp = self.undistort(measurementVector[i],Kp,Dp[0],Dp[1],Dp[2],Dp[3],Dp[4])

            measurementList.append(self.__measure(Kc,iKc,Kp,R,t,A,pixelVectorTemp,measurementVectorTemp).T)
        
        #Measurement function        
        return measurementList


#%%
class InputParameter(Common):  
    """
    ------------| fx
    Camera      | fy
    matrix      | s
                | u0
                | v0
    ------------| k1
    Camera      | k2
    distortion  | k3
                | p1
                | p2
    ------------| fx
    Projector   | fy
    matrix      | s
                | u0
                | v0
    ------------| k1
    Projector   | k2
    distortion  | k3
                | p1
                | p2
    ------------| Tx
    Projector   | Ty
    rotation    | Tz
    ------------| tx
    Projector   | ty
    translation | tz
    ------------
    
    """
    
    def __init__(self,useCUDA = False):

        if useCUDA: import cupy as xp
        else: import numpy as xp
        
        self.useCUDA = useCUDA
        self.xp = xp
                
        self.Kc = None
        self.Dc = None
        
        self.Kp = None
        self.Dp = None
        
        self.t = None
        self.r = None

    def setParameters(self,camArray, projArray, extArray):
        
        self.Kc = self.assembleCameraMatrix(
            camArray[0],
            camArray[1],
            camArray[2],
            camArray[3],
            camArray[4])
        self.Kp = self.assembleCameraMatrix(
            projArray[0],
            projArray[1],
            projArray[2],
            projArray[3],
            projArray[4])
        
        self.Dc = camArray[5:10].flatten()
        self.Dp = projArray[5:10].flatten()
        
        self.t = extArray[3:].reshape(-1,1)
        self.r = extArray[:3].flatten()
        

    def loadParameters(self,filename):
            
        with h5py.File(filename, 'r') as f:
            
            camArray = f["/camera/array"][()].flatten()
            projArray = f["/projector/array"][()].flatten()
            extArray = f["/extrinsic/array"][()].flatten()
            
            cov = f["covariance"][()]
        

        self.setParameters(camArray,projArray,extArray)
        
        self.setCovariance(cov)  
        
    def get(self):
        R = self.rodrigues(self.r[0], self.r[1], self.r[2])
        return self.Kc,self.Dc,self.Kp,self.t,R,self.Dp
    
    def getVirtual(self):
        
        xp = self.xp
        
        R = self.C @ xp.random.randn(self.C.shape[0])
        
        Kc = self.Kc + self.assembleCameraMatrix(R[0],R[1],R[2],R[3],R[4])
        Kc[2,2] = 1
        Dc = self.Dc + R[5:10]
        
        Kp = self.Kp + self.assembleCameraMatrix(R[11],R[12],R[13],R[14],R[15])
        Kp[2,2] = 1
        Dp = self.Dp + R[15:20]
        
        Rout = self.rodrigues(self.r[0] + R[20], self.r[1] + R[21], self.r[2] + R[22]) 
        
        t = self.t + R[23:26].reshape(-1,1)
        
        return Kc,Dc,Kp,t,Rout,Dp


        
        
            
        
        