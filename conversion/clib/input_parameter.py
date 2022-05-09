import h5py
from commonlib.rotations import rodrigues

class InputParameter():  
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
                | u0
                | v0
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
        
        self.nK = 5
        self.nD = 7
        self.nI = self.nK + self.nD
        
    def assemble_camera_matrix(self,fx,fy,s,u0,v0):
        xp = self.xp
        
        out = xp.eye(3)
        out[0,0] = fx 
        out[1,1] = fy
        out[0,1] = s
        out[0,2] = u0
        out[1,2] = v0
        
        return out
        
    def set_parameters(self,camArray, projArray, extArray):
        
        self.Kc = self.assemble_camera_matrix(
            camArray[0],
            camArray[1],
            camArray[2],
            camArray[3],
            camArray[4])
        self.Kp = self.assemble_camera_matrix(
            projArray[0],
            projArray[1],
            projArray[2],
            projArray[3],
            projArray[4])
        
        self.Dc = camArray[5:self.nI].flatten()
        self.Dp = projArray[5:self.nI].flatten()
        
        self.t = extArray[3:].reshape(-1,1)
        self.r = extArray[:3].flatten()

    def load_parameters(self,filename):
                
        with h5py.File(filename, 'r') as f:
            
            camArray = f["/camera/array"][()].flatten()
            projArray = f["/projector/array"][()].flatten()
            extArray = f["/extrinsic/array"][()].flatten()
            
            cov = f["covariance"][()]
        
        self.set_parameters(camArray,projArray,extArray)

        self.set_covariance(cov) 
        
    def get(self):
        R = rodrigues(self.r[0], self.r[1], self.r[2])
        return self.Kc,self.Dc,self.Kp,self.Dp,R,self.t
    
    def getVirtual(self):
        
        xp = self.xp
        
        R = self.C @ xp.random.randn(self.C.shape[0])
        
        Kc = self.Kc + self.assembleCameraMatrix(R[0],R[1],R[2],R[3],R[4])
        Kc[2,2] = 1
        Dc = self.Dc + R[5:12]
        
        Kp = self.Kp + self.assembleCameraMatrix(R[12],R[13],R[14],R[15],R[16])
        Kp[2,2] = 1
        Dp = self.Dp + R[17:24]
        
        Rout = rodrigues(self.r[0] + R[24], self.r[1] + R[25], self.r[2] + R[26]) 
        
        t = self.t + R[27:30].reshape(-1,1)
                
        return Kc,Dc,Kp,Dp,t,Rout
    
    
    def set_covariance(self,covIn):
        xp = self.xp
        self.cov = covIn
        self.C = xp.linalg.cholesky(self.cov)

        
        