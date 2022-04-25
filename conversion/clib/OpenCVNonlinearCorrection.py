import numpy as np
import matplotlib.pyplot as plt
import cv2
from clib.Common import Common

class NonlinearCorrection(Common):
    
    def __init__(self, useCUDA = False):
        
        if useCUDA: import cupy as xp
        else: import numpy as xp
        self.xp = xp
        self.useCUDA = useCUDA
    
    def distort(self,data,K,k1,k2,k3,p1,p2):
                
        xp = self.xp
        
        data = self.toWorld(data,K)
        
        x = data[0,:]
        y = data[1,:]
        
        r2 = x**2 + y**2    
        xy = x*y
                
        out = xp.empty_like(data)
        
        out[0,:] = x * (1 + k1*r2 + k2*r2**2 + k3*r2**3) + 2*p1*xy + p2*(r2 + 2*x**2)
        out[1,:] = y * (1 + k1*r2 + k2*r2**2 + k3*r2**3) + 2*p2*xy + p1*(r2 + 2*y**2)
        out[2,:] = 1
        
        out = self.toCamera(out,K)
                
        return out
    
    def undistort(self,data,K,k1,k2,k3,p1,p2):
        
        xp = self.xp
        
        get = self.get
        go = self.go
                
        data = self.toWorld(data,K)
        
        dCoeff = np.array([get(k1),get(k2),get(p1),get(p2),get(k3)])
                        
        undistortedData = xp.empty_like(data)
        undistortedData[:2,:] = go(cv2.undistortPoints(get(data[:2,:]), np.eye(3), dCoeff)[:,0,:].T)
        undistortedData[2,:] = 1
                
        undistortedData = self.toCamera(undistortedData,K)
        
        return undistortedData
    
    def toWorld(self,data,K):
        xp = self.xp
        
        Kinv = xp.linalg.inv(K)
        
        out = Kinv@data
        
        return out
        
    def toCamera(self,data,K):
        
        out = K@data
        
        return out

    def testDistortion(self,K,k1,k2,k3,p1,p2,res = (5120,5120)):
        xp = self.xp
        get = self.get
        u,v = xp.meshgrid(xp.linspace(1,res[0],res[0]//100), xp.linspace(1,res[1],res[1]//100), indexing = 'ij')
        u=u.flatten()
        v=v.flatten()
               
        vec = xp.empty((3,u.shape[0]))
        vec[0,:] = u
        vec[1,:] = v
        vec[2,:] = 1
                
        vecD = self.distort(vec,K,k1,k2,k3,p1,p2)
                
        plt.figure()
        plt.scatter(get(u),get(v),s=1,c='k')
        plt.scatter(get(vecD[0,:]),get(vecD[1,:]),s=1,c='r')
        plt.title('Distorted vectors')
        
        plt.figure()
        plt.quiver(get(u),get(v),-get(u)+get(vecD[0,:]),-get(v)+get(vecD[1,:]))
        
        print(xp.sum( (vec-vecD)**2)**0.5)
        
    def testUndistortion(self,K,k1,k2,k3,p1,p2,res = (5120,5120)):
        xp = self.xp
        get = self.get
        u,v = xp.meshgrid(xp.linspace(1,res[0],res[0]//100), xp.linspace(1,res[1],res[1]//100), indexing = 'ij')
        u=u.flatten()
        v=v.flatten()
        
        vec = xp.empty((3,u.shape[0]))
        vec[0,:] = u
        vec[1,:] = v
        vec[2,:] = 1
                
        vecD = self.distort(vec,K,k1,k2,k3,p1,p2)
        vecUD = self.undistort(vecD,K,k1,k2,k3,p1,p2)

        plt.figure()
        plt.scatter(get(vec[0,:]),get(vec[1,:]),s=1,c='k')
        plt.scatter(get(vecUD[0,:]),get(vecUD[1,:]),s=1,c='r')
        
        print(xp.sum( (vec-vecD)**2)**0.5)
        print(xp.sum( (vec-vecUD)**2)**0.5)
                

      
