import matplotlib.pyplot as plt
import time

class NonlinearCorrection():
    
    def __init__(self, useCUDA = False):
        
        if useCUDA: import cupy as xp
        else: import numpy as xp
        self.xp = xp
        self.useCUDA = useCUDA
    
    def distort5(self,data,K,Kinv,D):
        k1 = D[0]
        k2 = D[1]
        k3 = D[2]
        p1 = D[3]
        p2 = D[4]

        xp = self.xp
        
        # data = Kinv@data
        
        x = data[0,:]
        y = data[1,:]
        
        r2 = x**2 + y**2    
        xy = x*y
                
        out = xp.ones_like(data)
        
        out[0,:] = x * (1 + k1*r2 + k2*r2**2 + k3*r2**3) + 2*p1*xy + p2*(r2 + 2*x**2)
        out[1,:] = y * (1 + k1*r2 + k2*r2**2 + k3*r2**3) + 2*p2*xy + p1*(r2 + 2*y**2)

        # out = K@out
                
        return out
    
    def distort7(self,data,K,Kinv,D):
        k1 = D[0]
        k2 = D[1]
        k3 = D[2]
        p1 = D[3]
        p2 = D[4]
        x0 = D[5]
        y0 = D[6]
        xp = self.xp
        
        data = Kinv@data
        
        x = data[0,:] - x0
        y = data[1,:] - y0
        
        r2 = x**2 + y**2    
        xy = x*y
                
        out = xp.ones_like(data)
        
        out[0,:] = x * (1 + k1*r2 + k2*r2**2 + k3*r2**3) + 2*p1*xy + p2*(r2 + 2*x**2) + D[5]
        out[1,:] = y * (1 + k1*r2 + k2*r2**2 + k3*r2**3) + 2*p2*xy + p1*(r2 + 2*y**2) + D[6]

        out = K@out
                
        return out
        
    def undistort5(self,data,K,Kinv,D,maxIter=100,minDist=1e-7):
        xp = self.xp
        
        index = xp.ones(data.shape[1], dtype = bool)
        
        data = Kinv@data

        guess = data.copy()
        
        #First         
        distortGuess = self.distort5(guess[:,index],K,Kinv,D)
        
        diff = data[:,index] - distortGuess
           
        guess[:,index] += diff
        
        index[index] = xp.sum(diff**2,axis=0)>minDist
            
        i=0
        while xp.any(index):
        
            distortGuess[:,index] = self.distort5(guess[:,index],K,Kinv,D)
            
            diff = data[:,index] - distortGuess[:,index]
                       
            guess[:,index] += diff
            
            index[index] = xp.sum(diff**2,axis=0)>minDist
            
            if i>maxIter:break
            
            i+=1
           
        # error = xp.sum((K@data - K@distortGuess)**2)**0.5

        guess = K@guess
        
        return guess
    
    def undistort7(self,data,K,Kinv,D,maxIter=100000,minDist=1e-12):
        xp = self.xp
        
        index = xp.ones(data.shape[1], dtype = bool)
        
        data = Kinv@data

        guess = data.copy()
        
        #First         
        distortGuess = self.distort7(guess[:,index],xp.eye(3),xp.eye(3),D)
        
        diff = data[:,index] - distortGuess
           
        guess[:,index] += diff
        
        index[index] = xp.sum(diff**2,axis=0)>minDist
            
        i=0
        while xp.any(index):
        
            distortGuess[:,index] = self.distort7(guess[:,index],xp.eye(3),xp.eye(3),D)
            
            diff = data[:,index] - distortGuess[:,index]
                       
            guess[:,index] += diff
            
            index[index] = xp.sum(diff**2,axis=0)>minDist
            
            if i>maxIter:break
            
            i+=1
           
        # error = xp.sum((K@data - K@distortGuess)**2)**0.5

        guess = K@guess
        
        return guess
    
    def testDistortion(self,K,Kinv,res,D):
        xp = self.xp
        get = self.get
        u,v = xp.meshgrid(xp.linspace(1,res[0],res[0]//32), xp.linspace(1,res[1],res[1]//32), indexing = 'ij')
        u=u.flatten()
        v=v.flatten()
               
        vec = xp.empty((3,u.shape[0]))
        vec[0,:] = u
        vec[1,:] = v
        vec[2,:] = 1
    
        vec = Kinv@vec
        vecD = self.distort7(vec,K,Kinv,D)
        vecD = K@vecD
        vec = K@vec
        
        plt.figure()
        plt.scatter(get(u),get(v),s=1,c='k')
        plt.scatter(get(vecD[0,:]),get(vecD[1,:]),s=1,c='r')
        plt.title('Distorted vectors')
        
        plt.figure()
        plt.quiver(get(u),get(v),-get(u)+get(vecD[0,:]),-get(v)+get(vecD[1,:]))
        
        print(xp.sum( (vec-vecD)**2)**0.5)
        
    def testUndistortion(self,K,Kinv,res,D,minDist = 1e-7, downScale=32):
        
        xp = self.xp
        get = self.get
        u,v = xp.meshgrid(xp.linspace(1,res[0],res[0]//downScale), xp.linspace(1,res[1],res[1]//downScale), indexing = 'ij')
        u=u.flatten()
        v=v.flatten()
        
        vec = xp.empty((3,u.shape[0]))
        vec[0,:] = u
        vec[1,:] = v
        vec[2,:] = 1
        
        vec = Kinv@vec
        vecD = self.distort5(vec,K,Kinv,D)
        vecD = K@vecD
        vec = K@vec
        
        t1=time.time()
        vecUD, error = self.undistort5(vecD,K,Kinv,D,minDist = minDist)
        print(time.time()-t1)
        # plt.figure()
        # plt.scatter(get(vec[0,:]),get(vec[1,:]),s=1,c='k')
        # plt.scatter(get(vecUD[0,:]),get(vecUD[1,:]),s=1,c='r')
        
        # print(xp.sum( (vec-vecD)**2)**0.5)
        print(xp.sum( (vec-vecUD)**2)**0.5)
        # print(error)
      
