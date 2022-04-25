import os


class Common():
    
    def __init__(self, useCUDA):
        
        self.useCUDA = useCUDA
    
    def convertCoordsToMap(self,x,y,value,mapShape):
        xp = self.xp
        
        assert x.max()<mapShape[0]
        assert y.max()<mapShape[1]
        
        if xp.shape(value)==():
            output = xp.zeros(mapShape, dtype=type(value))
        else:
            output = xp.zeros(mapShape, dtype=value.dtype)    
            
        output[x, y] = value
        return output
    
    def convertCartToSph(self,points):
        xp = self.xp

        output = xp.empty_like(points)
        
        output[:,0] = xp.sum(points**2, axis=1)**0.5
        output[:,1] = xp.arctan2(points[:,0], points[:,1])
        output[:,2] = xp.arccos(points[:,2]/output[:,0])
        
        return output
        
    def createRotationMatrix(self,u,T):
        
        xp = self.xp
        
        x = u.flatten()[0]
        y = u.flatten()[1]
        z = u.flatten()[2]

        R = xp.empty((3,3))
        R[0,0] = xp.cos(T) + x**2*(1-xp.cos(T)) 
        R[0,1] = x*y*(1-xp.cos(T)) - z*xp.sin(T)
        R[0,2] = x*z*(1-xp.cos(T)) + y*xp.sin(T)
        R[1,0] = y*x*(1-xp.cos(T)) + z*xp.sin(T)
        R[1,1] = xp.cos(T) + y**2*(1-xp.cos(T))
        R[1,2] = y*z*(1-xp.cos(T)) - x*xp.sin(T)
        R[2,0] = z*x*(1-xp.cos(T)) - y*xp.sin(T)
        R[2,1] = z*y*(1-xp.cos(T)) + x*xp.sin(T)
        R[2,2] = xp.cos(T) + z**2*(1-xp.cos(T))
        
        return R
    
    def go(self,array):
        
        if self.useCUDA:
            xp=self.xp
            return xp.array(array)
        else:return array
    
    def get(self,array):
        
        if self.useCUDA:
            xp=self.xp
            return xp.asnumpy(array)
        else:return array
    
    def incrementMeanVar(self, oldVar, oldMean, newObs, N):
                
        newMean = oldMean + (newObs - oldMean)/N
        
        newVar = oldVar + (newObs - oldMean)*(newObs - newMean)
        
        return newMean, newVar
    
    def rodrigues(self,r1, r2, r3):
        xp = self.xp
                    
        theta = (r1**2 + r2**2 + r3**2)**0.5
               
        K = xp.zeros((3,3))
        K[0,1] = -r3
        K[0,2] = r2
        K[1,0] = r3
        K[1,2] = -r1
        K[2,0] = -r2
        K[2,1] = r1
        
        K = K/theta

        R = xp.eye(3) + xp.sin(theta)*K + (1-xp.cos(theta))*K@K
        
        return R
    
    def assembleCameraMatrix(self,fx,fy,s,u0,v0):
        xp = self.xp
        K = xp.eye(3)
        K[0,0] = fx
        K[1,1] = fy
        K[0,1] = s
        K[0,2] = u0
        K[1,2] = v0
        
        return K
    
    def setCovariance(self,covIn):
        xp = self.xp
        self.cov = covIn
        self.C = xp.linalg.cholesky(self.cov)
        
    def setParameters(self, **kwargs):
        
        for key in kwargs:
            assert hasattr(self,key)
        
        for key in kwargs:
            setattr(self, key, kwargs[key]) 
            
def selectModel(model):
    
    currentDir = os.getcwd()
    cutoff = currentDir.find("ProjectyBoy2000")
    assert cutoff!=-1
    home = currentDir[:cutoff+16]
    
    if model == 'modelA':
        parameterFilename = home + r"\Calibration\Temp Data\ModelA_Parameters5K.hdf5"
        from mlib.modelA import Measurement
    elif model == 'modelB':
        parameterFilename = home + r"\Calibration\Temp Data\ModelB_Parameters.hdf5"
        from mlib.modelB import Measurement
    elif model == 'modelC':
        parameterFilename = home + r"\Calibration\Temp Data\ModelC_Parameters1K.hdf5"
        from mlib.modelC import Measurement
    elif model == 'modelD':
        print('not done')
        parameterFilename = home + r"\Calibration\Temp Data\ModelC_Parameters1K.hdf5"
        from mlib.modelC import Measurement
    elif model == 'modelE':
        parameterFilename = home + r"\Calibration\Temp Data\ModelE_Parameters.hdf5"
        from mlib.modelA import Measurement
    else:
        raise Exception
    
    return Measurement, parameterFilename


            
            