from clib.Common import Common
import h5py

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

    def loadParameters(self,filename):
        
        go = self.go
        
        with h5py.File(filename, 'a') as f:
            
            Kc = f["/camera/K"][()]
            Dc = f["/camera/D"][()]
            Kp = f["/projector/K"][()]
            Dp = f["/projector/D"][()]
            r =  f["/extrinsic/r"][()]
            t =  f["/extrinsic/t"][()]
            
            cov = f["covariance"][()]
            
            
        self.setParameters(
            Kc = go(Kc),
            Kp = go(Kp),
            Dc = go(Dc),
            Dp = go(Dp),
            t = go(t).reshape(-1,1),
            r = go(r)
            )
        
        self.setCovariance(go(cov))   
        
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
        safsdf
        
        return Kc,Dc,Kp,t,Rout,Dp


        
        