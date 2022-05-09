import h5py


from clib.input_measurement import InputMeasurement
from clib.input_parameter import InputParameter
from clib.nonlinear_correction import NonlinearCorrection

#%%
class Measurement():
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
        self.undistort = self.inputC.undistort7
        self.distort = self.inputC.distort7
        
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
        Kc,Dc,Kp,Dp,R,t = self.inputP.get()
        A = self.A
        
        pixelVector, measurementVector = self.inputM.get()      
        
        iKc = xp.linalg.inv(Kc)
        iKp = xp.linalg.inv(Kp)

        measurementList = []
        
        
        for i in range(len(pixelVector)):
    
            pixelVectorTemp = self.undistort(pixelVector[i],Kc,iKc,Dc)
            measurementVectorTemp = self.undistort(measurementVector[i],Kp,iKp,Dp)

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



        
        
            
        
        