import tensorflow as tf
import numpy as np
import h5py

DATATYPE = tf.float64

from clib.BaseCalibration import Calibration
from clib.BaseData import DataTF
from commonlib.console_outputs import ProgressBar
from commonlib.h5py_functions import num_of_keys


#%% Calibration
class NonLinearCalibration(Calibration):
    
    def __init__(self, numCPoints, numPPoints, numPositions):
        super().__init__()
        
        self.nK = 5
        self.nD = 7
        self.nI = self.nK + self.nD
        
        numPoints = numCPoints + numPPoints
        
        self.weightedOptimise = tf.function(self.weightedTrain, 
                input_signature=(
                    tf.TensorSpec(shape=[3,numCPoints], dtype=DATATYPE),
                    tf.TensorSpec(shape=[3,numPPoints], dtype=DATATYPE),
                    tf.TensorSpec(shape=[numPoints*numPositions*2,1], dtype=DATATYPE),
                    tf.TensorSpec(shape=[self.nI*2 + 6 + numPositions*6], dtype=DATATYPE),
                    tf.TensorSpec(shape=[numPoints*numPositions*2,numPoints*numPositions*2], dtype=DATATYPE),
                    tf.TensorSpec(shape=[], dtype=DATATYPE),
                    tf.TensorSpec(shape=[], dtype=tf.int32),
                    tf.TensorSpec(shape=[], dtype=tf.int32),
                    tf.TensorSpec(shape=[], dtype=DATATYPE),
                    tf.TensorSpec(shape=[], dtype=tf.int32),
                    ))
        
        # self.Optimise = tf.function(self.train, 
        #         input_signature=(
        #             tf.TensorSpec(shape=[3,numCPoints], dtype=DATATYPE),
        #             tf.TensorSpec(shape=[3,numPPoints], dtype=DATATYPE),
        #             tf.TensorSpec(shape=[numPoints*numPositions*2,1], dtype=DATATYPE),

        #             tf.TensorSpec(shape=[self.nI*2 + 6 + numPositions*6], dtype=DATATYPE),
                    
        #             tf.TensorSpec(shape=[numPoints*numPositions*2,numPoints*numPositions*2], dtype=DATATYPE),
                    
        #             tf.TensorSpec(shape=[], dtype=DATATYPE),
        #             tf.TensorSpec(shape=[], dtype=tf.int32),
        #             tf.TensorSpec(shape=[], dtype=tf.int32),
        #             tf.TensorSpec(shape=[], dtype=DATATYPE),
        #             tf.TensorSpec(shape=[], dtype=tf.int32),
        #             ))
 
    @tf.function(input_signature=(tf.TensorSpec(shape=[3,None], dtype=DATATYPE),
                              tf.TensorSpec(shape=[3,3], dtype=DATATYPE),
                              tf.TensorSpec(shape=[3,3], dtype=DATATYPE),
                              tf.TensorSpec(shape=[3,1], dtype=DATATYPE),
                              tf.TensorSpec(shape=[None], dtype=DATATYPE),
                              ))
    def backProject(self,x,K,R,T,D):
        
        print("Tracing: backProject")
        
        estimate = R @ x + T
        
        estimate = estimate/estimate[2,:]
        
        estimate = self.distort(estimate, D)
        
        estimate = K @ estimate
        
        return estimate[:2,:]
                    
    @tf.function
    def transformFunction(self,cPoints,pPoints,X):
        
        print("Tracing: transformFunction")
        
        nK = self.nK
        nI = self.nI
        
        kc = X[:nK]
        Dc = X[nK:nI]
        kp = X[nI:nI+nK]
        Dp = X[nI+nK:2*nI]
        Q = X[2*nI:2*nI+6]
        e_all = X[2*nI+6:]
        
        Kc = self.cameraMatrix(kc)
        Kp = self.cameraMatrix(kp)
        
        r,t = tf.split(Q, [3,3])
        Rp = self.rodrigues(r)
        Tp = tf.reshape(t, (3,1)) 
                
        extrinsics = tf.split(e_all, e_all.shape[0]//6)   
        
        #obtain grad
        r,t = tf.split(extrinsics[0], [3,3])
        R = self.rodrigues(r)
        T = tf.reshape(t, (3,1)) 
        
        camEst = R @ cPoints + T
        projEst =  Rp @ (R @ pPoints + T) + Tp
        
        for i in range(1,len(extrinsics)):
                                           
            #obtain grad
            r,t = tf.split(extrinsics[i], [3,3])
            R = self.rodrigues(r)
            T = tf.reshape(t, (3,1)) 
            
            camEst = tf.concat((camEst, R @ cPoints + T),axis=1)
            projEst = tf.concat((projEst, Rp @ (R @ pPoints + T) + Tp),axis=1)
                
        camEst = camEst/camEst[2,:]
        projEst = projEst/projEst[2,:]
        
        camEst = self.distort(camEst, Dc)
        projEst = self.distort(projEst, Dp)
        
        camEst = Kc @ camEst
        projEst = Kp @ projEst
        
        camEst = tf.reshape(tf.transpose(camEst[:2,:]), (-1,1))
        projEst = tf.reshape(tf.transpose(projEst[:2,:]), (-1,1))

        allEst = tf.concat((camEst,projEst), axis=0)        

        return allEst 
                
    @tf.function(input_signature=(tf.TensorSpec(shape=[3,None], dtype=DATATYPE),
                                  tf.TensorSpec(shape=[3,None], dtype=DATATYPE),
                                  tf.TensorSpec(shape=[None], dtype=DATATYPE),
                                  ))
    def gradFunction(self,cPoints,pPoints,X):
        print("Tracing: gradFunction")
        nK = self.nK
        nI = self.nI
        
        #Set up inputs
        kc = X[:nK]
        Dc = X[nK:nI]
        kp = X[nI:nI+nK]
        Dp = X[nI+nK:2*nI]
        Q = X[2*nI:2*nI+6]
        e = X[2*nI+6:]
            
        with tf.GradientTape(persistent = False) as tape:
            
            tape.watch(kc)
            tape.watch(Dc)            
            tape.watch(e)
            
            r,t = tf.split(e, [3,3])
            R = self.rodrigues(r)
            T = tf.reshape(t, (3,1)) 
                           
            Kc = self.cameraMatrix(kc)
            camEst = R @ cPoints + T
            
            camEst = camEst/camEst[2,:]
            
            camEst = self.distort(camEst, Dc)

            camEst = Kc @ camEst

            camEst = tf.reshape(tf.transpose(camEst[:2,:]), (-1,1))[:,0]
        
        camGrad = tape.jacobian(camEst, [kc,Dc,e])
            
        with tf.GradientTape(persistent = False) as tape:

            tape.watch(kp)
            tape.watch(Dp)
            tape.watch(e)
            tape.watch(Q)
            
            #obtain extrinsics
            r,t = tf.split(e, [3,3])
            R = self.rodrigues(r)
            T = tf.reshape(t, (3,1)) 
            
            rp,tp = tf.split(Q, [3,3])
            Rp = self.rodrigues(rp)
            Tp = tf.reshape(tp, (3,1)) 
            
            Kp = self.cameraMatrix(kp)
            
            projEst = Rp @ (R @ pPoints + T) + Tp
            projEst = projEst/projEst[2,:]
            projEst = self.distort(projEst, Dp)
            projEst = Kp @ projEst
            
            projEst = tf.reshape(tf.transpose(projEst[:2,:]), (-1,1))[:,0]

        projGrad = tape.jacobian(projEst, [kp,Dp,Q,e])
                
        return camGrad,projGrad
    
    @tf.function
    def jacobianFunction(self,cPoints,pPoints,X):
        
        nK = self.nK
        nD = self.nD
        nI = self.nI
        
        print("Tracing: jacobianFunction")
        
        consistent = X[:2*nI+6]
        extrinsicsAll = X[2*nI+6:]
        
        numPositions = extrinsicsAll.shape[0]//6
        numCPoints = cPoints.shape[1]//1
        numPPoints = pPoints.shape[1]//1
        numPoints = numCPoints + numPPoints
    
        extrinsics = tf.split(extrinsicsAll, numPositions)   
        
        jacobian = tf.zeros((numPoints*numPositions*2, numPositions*6 + 2*nI + 6),dtype=DATATYPE)
        
        #create base indices for scattering updates into jacobian with
        i,j = tf.meshgrid(tf.linspace(0,numCPoints*2-1,numCPoints*2),tf.linspace(0,nK-1,nK), indexing='ij')
        i = tf.cast(tf.reshape(i, (numCPoints*5*2,1)), dtype = tf.int32)
        j = tf.cast(tf.reshape(j, (numCPoints*5*2,1)), dtype = tf.int32)
        cKIndex = tf.concat((i,j),axis=1)
                
        i,j = tf.meshgrid(tf.linspace(0,numCPoints*2-1,numCPoints*2),tf.linspace(0,nD-1,nD), indexing='ij')
        i = tf.cast(tf.reshape(i, (numCPoints*nD*2,1)), dtype = tf.int32)
        j = tf.cast(tf.reshape(j, (numCPoints*nD*2,1)), dtype = tf.int32) + nK
        cDIndex = tf.concat((i,j),axis=1)
        
        i,j = tf.meshgrid(tf.linspace(0,numCPoints*2-1,numCPoints*2),tf.linspace(0,5,6), indexing='ij')
        i = tf.cast(tf.reshape(i, (numCPoints*6*2,1)), dtype = tf.int32)
        j = tf.cast(tf.reshape(j, (numCPoints*6*2,1)), dtype = tf.int32) + 2*nI + 6
        cEIndex = tf.concat((i,j),axis=1)
        
        
        
        i,j = tf.meshgrid(tf.linspace(0,numPPoints*2-1,numPPoints*2),tf.linspace(0,nK-1,nK), indexing='ij')
        i = tf.cast(tf.reshape(i, (numPPoints*5*2,1)), dtype = tf.int32) + numCPoints*2*numPositions
        j = tf.cast(tf.reshape(j, (numPPoints*5*2,1)), dtype = tf.int32) + nI
        pKIndex = tf.concat((i,j),axis=1)
        
        i,j = tf.meshgrid(tf.linspace(0,numPPoints*2-1,numPPoints*2),tf.linspace(0,nD-1,nD), indexing='ij')
        i = tf.cast(tf.reshape(i, (numPPoints*nD*2,1)), dtype = tf.int32) + numCPoints*2*numPositions
        j = tf.cast(tf.reshape(j, (numPPoints*nD*2,1)), dtype = tf.int32) + nI + nK
        pDIndex = tf.concat((i,j),axis=1)
        
        i,j = tf.meshgrid(tf.linspace(0,numPPoints*2-1,numPPoints*2),tf.linspace(0,5,6), indexing='ij')
        i = tf.cast(tf.reshape(i, (numPPoints*6*2,1)), dtype = tf.int32) + numCPoints*2*numPositions
        j = tf.cast(tf.reshape(j, (numPPoints*6*2,1)), dtype = tf.int32) + 2*nI + 6
        pEIndex = tf.concat((i,j),axis=1)
        
        i=0
        for extrinsic in extrinsics:
            
            #parameter set for each position
            parameters = tf.concat((consistent,extrinsic),axis=0)
            
            #constants to be added for each iteration
            cAdd = i*numCPoints*2
            pAdd = i*numPPoints*2
            rAdd = i*6
            
            #obtain grad
            camGrad, projGrad = self.gradFunction(cPoints,pPoints,parameters)
            
            #camera
            kGrad = tf.reshape(camGrad[0], (numCPoints*nK*2,1))[:,0]
            jacobian = tf.tensor_scatter_nd_update(
                jacobian, 
                cKIndex + tf.constant((cAdd,0), dtype=tf.int32), 
                kGrad
                )
            DGrad = tf.reshape(camGrad[1], (numCPoints*nD*2,1))[:,0]
            jacobian = tf.tensor_scatter_nd_update(
                jacobian, 
                cDIndex + tf.constant((cAdd,0), dtype=tf.int32), 
                DGrad
                )
            eGrad = tf.reshape(camGrad[2], (numCPoints*6*2,1))[:,0]
            jacobian = tf.tensor_scatter_nd_update(
                jacobian, 
                cEIndex + tf.constant((cAdd,rAdd), dtype=tf.int32), 
                eGrad
                )
            
            #Projector
            kGrad = tf.reshape(projGrad[0], (numPPoints*nK*2,1))[:,0]
            jacobian = tf.tensor_scatter_nd_update(
                jacobian, 
                pKIndex + tf.constant((pAdd,0), dtype=tf.int32), 
                kGrad
                )
            DGrad = tf.reshape(projGrad[1], (numPPoints*nD*2,1))[:,0]
            jacobian = tf.tensor_scatter_nd_update(
                jacobian, 
                pDIndex + tf.constant((pAdd,0), dtype=tf.int32), 
                DGrad
                )
            QGrad = tf.reshape(projGrad[2], (numPPoints*6*2,1))[:,0]
            jacobian = tf.tensor_scatter_nd_update(
                jacobian, 
                pEIndex + tf.constant((pAdd,-6), dtype=tf.int32), 
                QGrad
                )
            eGrad = tf.reshape(projGrad[3], (numPPoints*6*2,1))[:,0]
            jacobian = tf.tensor_scatter_nd_update(
                jacobian, 
                pEIndex + tf.constant((pAdd,rAdd), dtype=tf.int32), 
                eGrad
                )
            
            i=i+1
    
        return jacobian
             
    @tf.function(input_signature=(tf.TensorSpec(shape=[3,None], dtype=DATATYPE),
                          tf.TensorSpec(shape=[None], dtype=DATATYPE),
                          ))
    def distort(self,x,X):
                
        k1 = X[0]
        k2 = X[1]
        k3 = X[2]
        p1 = X[3]
        p2 = X[4]
        x0 = X[5]
        y0 = X[6]
        
        offset = tf.stack((x0,y0,tf.constant(1,DATATYPE)), axis=0)
        offset = tf.reshape(offset, (-1,1))
        
        x = x-offset
   
        r2 = tf.math.reduce_sum(x[:2,:]**2,axis=0,keepdims=True)
        
        w = tf.math.reduce_prod(x[:2,:],axis=0,keepdims=True)
        
        radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
        
        tangentialX = 2*p1*w + p2*(r2 + 2*x[0:1,:]**2)
        tangentialY = 2*p2*w + p1*(r2 + 2*x[1:2,:]**2)
        
        xDistort = tf.concat((
            x[0:1,:]*radial + tangentialX,
            x[1:2,:]*radial + tangentialY, 
            x[2:,:]
            ), axis=0)
        
        xDistort = xDistort+offset
   
        return xDistort
        
    @tf.function(input_signature=(
        tf.TensorSpec(shape=[3,1], dtype=DATATYPE),
        tf.TensorSpec(shape=[3,1], dtype=DATATYPE),
        tf.TensorSpec(shape=[None], dtype=DATATYPE),
        ))
    def singleHamiltonianFunction(self,xc,xp,X):
        
        print("Tracing: singleHamiltonianFunction")
        nK = self.nK
        nI = self.nI

                
        with tf.GradientTape(persistent = True) as tape1:
            tape1.watch(X)
            with tf.GradientTape(persistent = True) as tape2:
                
                tape2.watch(X)
                
                #Set up inputs
                kc = X[:nK]
                Dc = X[nK:nI]
                kp = X[nI:nI+nK]
                Dp = X[nI+nK:2*nI]
                Q = X[2*nI:2*nI+6]
                e = X[2*nI+6:]
                
                #obtain extrinsics
                r,t = tf.split(e, [3,3])
                R = self.rodrigues(r)
                T = tf.reshape(t, (3,1)) 
                
                rp,tp = tf.split(Q, [3,3])
                Rp = self.rodrigues(rp)
                Tp = tf.reshape(tp, (3,1)) 
                
                #obtain intrinsics       
                Kc = self.cameraMatrix(kc)
                Kp = self.cameraMatrix(kp)
                                
                cEstimate = R @ xc + T
            
                pEstimate = Rp @ (R @ xp + T) + Tp
            
                camEst = cEstimate/cEstimate[2,:]
                projEst = pEstimate/pEstimate[2,:]
                
                camEst = self.distort(camEst, Dc)
                projEst = self.distort(projEst, Dp)
    
                camEst = Kc @ camEst
                projEst = Kp @ projEst
                
                cy1 = camEst[0,0]
                cy2 = camEst[1,0]
                
                py1 = projEst[0,0]
                py2 = projEst[1,0]
                
            cdy1dx = tape2.gradient(cy1, X)
            cdy2dx = tape2.gradient(cy2, X)
            pdy1dx = tape2.gradient(py1, X)
            pdy2dx = tape2.gradient(py2, X)
        
        cd2y1dx2 = tape1.jacobian(cdy1dx, X)
        cd2y2dx2 = tape1.jacobian(cdy2dx, X)
        pd2y1dx2 = tape1.jacobian(pdy1dx, X)
        pd2y2dx2 = tape1.jacobian(pdy2dx, X)

        return cd2y1dx2,cd2y2dx2,pd2y1dx2,pd2y2dx2
    
    @tf.function()
    def blockHamiltonianFunction(self,xc,xp,X):
        
        print("Tracing: blockHamiltonianFunction")
        
        cNum = xc.shape[1]
        pNum = xp.shape[1]

        cH = tf.TensorArray(tf.float64, size=cNum*2)
        pH = tf.TensorArray(tf.float64, size=pNum*2)
        
        for i in tf.range(pNum):
            
            cd2y1dx2,cd2y2dx2,pd2y1dx2,pd2y2dx2 = self.singleHamiltonianFunction(xc[:,i:i+1],xp[:,i:i+1], X)
            
            cH = cH.write(2*i, cd2y1dx2)
            cH = cH.write(2*i+1, cd2y2dx2)
            
            pH = pH.write(2*i, pd2y1dx2)
            pH = pH.write(2*i+1, pd2y2dx2)
            
        for i in tf.range(pNum, cNum):
            
            cd2y1dx2,cd2y2dx2,pd2y1dx2,pd2y2dx2 = self.singleHamiltonianFunction(xc[:,i:i+1],xp[:,0:1], X)
            
            cH = cH.write(2*i, cd2y1dx2)
            cH = cH.write(2*i+1, cd2y2dx2)
            
        return cH.stack(),pH.stack()    
                
    def fullHamiltonianFunction(self,xc,xp,X):
                
        cN = xc.shape[1]
        pN = xp.shape[1]

        nK = self.nK
        nI = self.nI
           
        consistent = X[:2*nI+6]
        extrinsics = X[2*nI+6:]
        
        numPositions = extrinsics.shape[0]//6
        
        cTotal = cN*2*numPositions
        pTotal = pN*2*numPositions

        cH = np.zeros((36, 36, cTotal))
        pH = np.zeros((36, 36, pTotal))
        
        extrinsics = tf.split(extrinsics, numPositions)
        bar = ProgressBar()
        bar.updateBar(0,numPositions)
        
        for j in range(numPositions):
            
            #parameter set for each position
            parameters = tf.concat((consistent,extrinsics[j]),axis=0)
                        
            cHtemp,pHtemp = self.blockHamiltonianFunction(xc,xp,parameters)
            
            cHtemp = cHtemp.numpy()
            pHtemp = pHtemp.numpy()
            cHtemp = np.swapaxes(cHtemp, 0,2)
            pHtemp = np.swapaxes(pHtemp, 0,2)
            
            cH[:,:,j*cN*2:(j+1)*cN*2] = cHtemp
            pH[:,:,j*pN*2:(j+1)*pN*2] = pHtemp
            
            bar.updateBar(j+1, numPositions)
        H = np.concatenate((cH, pH), axis=2)
        return H
             
    @tf.function
    def boardGradFunction(self,cPoints,pPoints,X):        
        nK = self.nK
        nI = self.nI
        
        #Set up inputs
        kc = X[:nK]
        Dc = X[nK:nI]
        kp = X[nI:nI+nK]
        Dp = X[nI+nK:2*nI]
        Q = X[2*nI:2*nI+6]
        e = X[2*nI+6:]
        
        #obtain extrinsics
        r,t = tf.split(e, [3,3])
        R = self.rodrigues(r)
        T = tf.reshape(t, (3,1)) 
        
        rp,tp = tf.split(Q, [3,3])
        Rp = self.rodrigues(rp)
        Tp = tf.reshape(tp, (3,1)) 
        
        #obtain intrinsics       
        Kc = self.cameraMatrix(kc)
        Kp = self.cameraMatrix(kp)
        
        with tf.GradientTape(persistent = False) as tape:
                                                                            
            tape.watch(cPoints)
            
            camEst = R @ cPoints + T
            camEst = camEst/camEst[2,:]            
            camEst = self.distort(camEst, Dc)
            camEst = Kc @ camEst

            camEst = tf.reshape(tf.transpose(camEst[:2,:]), (-1,1))[:,0]
            
        camGrad = tape.jacobian(camEst, cPoints)
            
        with tf.GradientTape(persistent = False) as tape:

            tape.watch(pPoints)
            
            projEst = Rp @ (R @ pPoints + T) + Tp
            projEst = projEst/projEst[2,:]
            projEst = self.distort(projEst, Dp)
            projEst = Kp @ projEst
            projEst = tf.reshape(tf.transpose(projEst[:2,:]), (-1,1))[:,0]

        projGrad = tape.jacobian(projEst, pPoints)
                                    
        return camGrad, projGrad
    
    # @tf.function
    def boardJacobianFunction(self,cPoints,pPoints,X,cIndex,pIndex):
        
        print("Tracing: jacobianFunction")
        nK = self.nK
        nD = self.nD
        nI = self.nI

        intrinsics = X[:2*nI+6]
        extrinsicsAll = X[2*nI+6:]
        
        numPositions = extrinsicsAll.shape[0]//6
        numCPoints = cPoints.shape[1]//1
        numPPoints = pPoints.shape[1]//1
        numPoints = numCPoints+ numPPoints

        extrinsics = tf.split(extrinsicsAll, numPositions)   
        
        jacobian = tf.zeros((numPoints * numPositions * 2, numPoints*3),dtype=DATATYPE)
        
        #create base indices for scattering updates into jacobian with
        i,j = tf.meshgrid(tf.linspace(0,numCPoints*2-1,numCPoints*2),tf.linspace(0,numPoints-1,numPoints)*3, indexing='ij')
        i = tf.cast(tf.reshape(tf.gather(i,cIndex,axis=1), (numCPoints*2*numCPoints,1)), dtype = tf.int32)
        j = tf.cast(tf.reshape(tf.gather(j,cIndex,axis=1), (numCPoints*2*numCPoints,1)), dtype = tf.int32)
        baseCamIndices = tf.concat((i,j),axis=1)
        
        i,j = tf.meshgrid(tf.linspace(0,numPPoints*2-1,numPPoints*2),tf.linspace(0,numPoints-1,numPoints)*3, indexing='ij')
        i = tf.cast(tf.reshape(tf.gather(i,pIndex,axis=1), (numPPoints*2*numPPoints,1)), dtype = tf.int32) + numCPoints*numPositions*2
        j = tf.cast(tf.reshape(tf.gather(j,pIndex,axis=1), (numPPoints*2*numPPoints,1)), dtype = tf.int32)
        baseProjIndices = tf.concat((i,j),axis=1)
                
        i=0
        for extrinsic in extrinsics:
            
            #parameter set for each position
            parameters = tf.concat((intrinsics,extrinsic),axis=0)
            
            #constants to be added for each iteration
            cStart = i*numCPoints*2
            pStart = i*numPPoints*2
            
            #obtain grad
            camGrad, projGrad = self.boardGradFunction(cPoints,pPoints,parameters)
            
            #Camera
            xGrad = tf.reshape(camGrad[:,0,:], (numCPoints*2*numCPoints,1))[:,0]
            jacobian = tf.tensor_scatter_nd_update(
                jacobian, 
                baseCamIndices + tf.constant((cStart,0), dtype=tf.int32), 
                xGrad
                )
            yGrad = tf.reshape(camGrad[:,1,:], (numCPoints*2*numCPoints,1))[:,0]
            jacobian = tf.tensor_scatter_nd_update(
                jacobian, 
                baseCamIndices + tf.constant((cStart,1), dtype=tf.int32), 
                yGrad
                )
            zGrad = tf.reshape(camGrad[:,2,:], (numCPoints*2*numCPoints,1))[:,0]
            jacobian = tf.tensor_scatter_nd_update(
                jacobian, 
                baseCamIndices + tf.constant((cStart,2), dtype=tf.int32), 
                zGrad
                )
            
            #Projector
            xGrad = tf.reshape(projGrad[:,0,:], (numPPoints*2*numPPoints,1))[:,0]
            jacobian = tf.tensor_scatter_nd_update(
                jacobian, 
                baseProjIndices + tf.constant((pStart,0), dtype=tf.int32), 
                xGrad
                )
            yGrad = tf.reshape(projGrad[:,1,:], (numPPoints*2*numPPoints,1))[:,0]
            jacobian = tf.tensor_scatter_nd_update(
                jacobian, 
                baseProjIndices + tf.constant((pStart,1), dtype=tf.int32), 
                yGrad
                )
            zGrad = tf.reshape(projGrad[:,2,:], (numPPoints*2*numPPoints,1))[:,0]
            jacobian = tf.tensor_scatter_nd_update(
                jacobian, 
                baseProjIndices + tf.constant((pStart,2), dtype=tf.int32), 
                zGrad
                )
            
            i=i+1
    
        return jacobian
    
    def weightedTrain(self,cBoard,pBoard,points,params,W,dampingFactor,DISPLAY,FAILURE_COUNT_MAX,CHANGE_MIN,ITERATION_MAX):
        
        print("Tracing: Train")                
        optimised = tf.constant(False, dtype = tf.bool)
        
        failureCount = 0
                 
        epoch = tf.constant(0, dtype = tf.int32)
        
        #Initialise first loss and jacobian
        loss = W@(points - self.transformFunction(cBoard,pBoard,params))
        lossSum = tf.math.reduce_sum(loss**2)
        J = self.jacobianFunction(cBoard,pBoard,params)
        
        lossSumChange = tf.constant(1,dtype = DATATYPE)
                
        while not optimised:
            
            JtJ = self.weightedJtJFunction(J,W,dampingFactor)

            update = tf.linalg.inv(JtJ) @ tf.transpose(J) @ loss
                                               
            #Assign the update
            paramsUpdate = params + update[:,0]
            
            #Calculate a new loss
            lossUpdate = W@(points - self.transformFunction(cBoard,pBoard,paramsUpdate))
            
            #Has this improved the loss?
            lossSumUpdate = tf.reduce_sum(lossUpdate**2)
                                 
            lossSumChange = lossSumUpdate - lossSum
            
            condition = tf.math.less(lossSumChange, 0)

            #If condition is True
            if condition:
                 
                #Decrease the damping
                dampingFactor = self.iterateDamping(dampingFactor, 0.5)
                
                #Accept new value of loss, loss sum and parameters
                loss = lossUpdate
                lossSum = lossSumUpdate
                params = paramsUpdate

                #Calculate new jacobian
                J = self.jacobianFunction(cBoard,pBoard,params)
                              
                #Reset consecutive failure count
                failureCount = 0
                                                                
            #If condition2 fails    
            else:
                
                #Increase the damping
                dampingFactor = self.iterateDamping(dampingFactor, 5)
                failureCount = failureCount + 1
                                            
            #Optimisation Check
            optimised = (epoch>ITERATION_MAX) | (failureCount>FAILURE_COUNT_MAX) | ((lossSumChange<0)&(lossSumChange>-CHANGE_MIN))
            if DISPLAY == 1:self.printUpdate(epoch, dampingFactor, lossSum)
            
            epoch = tf.add(epoch, 1)

        if DISPLAY != 0: 
            self.printUpdate(epoch, dampingFactor, lossSum)            
            tf.print("\n===", "FINISHED", "===\n")
        
        loss = points - self.transformFunction(cBoard,pBoard,params)

        return params, J, loss
    
    def train(self,cBoard,pBoard,points,params,dampingFactor,DISPLAY,FAILURE_COUNT_MAX,CHANGE_MIN,ITERATION_MAX):
        
        print("Tracing: Train")                
        optimised = tf.constant(False, dtype = tf.bool)
        
        failureCount = 0
                 
        epoch = tf.constant(0, dtype = tf.int32)
        
        #Initialise first loss and jacobian
        loss = points - self.transformFunction(cBoard,pBoard,params)
        lossSum = tf.math.reduce_sum(loss**2)
        J = self.jacobianFunction(cBoard,pBoard,params)
        
        lossSumChange = tf.constant(1,dtype = DATATYPE)
                
        while not optimised:
            
            JtJ = self.JtJFunction(J,dampingFactor)

            update = tf.linalg.inv(JtJ) @ tf.transpose(J) @ loss
                                               
            #Assign the update
            paramsUpdate = params + update[:,0]
            
            #Calculate a new loss
            lossUpdate = points - self.transformFunction(cBoard,pBoard,paramsUpdate)
            
            #Has this improved the loss?
            lossSumUpdate = tf.reduce_sum(lossUpdate**2)
                                 
            lossSumChange = lossSumUpdate - lossSum
            
            condition = tf.math.less(lossSumChange, 0)

            #If condition is True
            if condition:
                 
                #Decrease the damping
                dampingFactor = self.iterateDamping(dampingFactor, 0.5)
                
                #Accept new value of loss, loss sum and parameters
                loss = lossUpdate
                lossSum = lossSumUpdate
                params = paramsUpdate

                #Calculate new jacobian
                J = self.jacobianFunction(cBoard,pBoard,params)
                              
                #Reset consecutive failure count
                failureCount = 0
                                                                
            #If condition2 fails    
            else:
                
                #Increase the damping
                dampingFactor = self.iterateDamping(dampingFactor, 5)
                failureCount = failureCount + 1
                                            
            #Optimisation Check
            optimised = (epoch>ITERATION_MAX) | (failureCount>FAILURE_COUNT_MAX) | ((lossSumChange<0)&(lossSumChange>-CHANGE_MIN))
            if DISPLAY == 1:self.printUpdate(epoch, dampingFactor, lossSum)
            
            epoch = tf.add(epoch, 1)

        if DISPLAY != 0: 
            self.printUpdate(epoch, dampingFactor, lossSum)            
            tf.print("\n===", "FINISHED", "===\n")
        
        loss = points - self.transformFunction(cBoard,pBoard,params)

        return params, J, loss
#%% InputData
class InputData(DataTF):
    
    def __init__(self):
        
        self.nD = 7
        self.nI = 5+self.nD
    
    def packParameters(self,Kc,Kp,Dc,Dp,tp,rp,t,r):
        nD = self.nD
        nI = self.nI
        
        #Assemble parameter vector
        X = np.empty((2*nI + 6 + 6*t.shape[1]))*np.nan
        
        X[0] = Kc[0,0]
        X[1] = Kc[1,1]
        X[2] = Kc[0,1]
        X[3] = Kc[0,2]
        X[4] = Kc[1,2]
        
        X[5:nI] = Dc.flatten()
        
        X[nI] = Kp[0,0]
        X[nI+1] = Kp[1,1]
        X[nI+2] = Kp[0,1]
        X[nI+3] = Kp[0,2]
        X[nI+4] = Kp[1,2]
        
        X[nI+5:2*nI] = Dc.flatten()
        
        X[2*nI:2*nI+3] = rp.flatten()
        X[2*nI+3:2*nI+6] = tp.flatten()
            
        for i in range(t.shape[1]):
            
            extrinsic = np.concatenate((r[:,i], t[:,i]), axis=0)
            
            X[i*6+2*nI+6:(i+1)*6+2*nI+6] = extrinsic
            
        return X
                        
    def loadBoardPoints(self, filename, I):
        
        with h5py.File(filename, 'r') as f:
            
            board = f["board"][:,:]
                   
        if board.shape[1] == 2:
            board = np.concatenate((board, np.zeros_like(board[:,0:1])), axis=1)
        board = board.astype(np.float64)
        
        self.setCameraBoardPoints(board[I,:])
        self.setProjectorBoardPoints(board[np.invert(I),:])
        
    def loadMeasuredPoints(self, filename, I):
                
        num_of_positions = num_of_keys(filename, "camera/points")
        
        cPoints = []
        pPoints = []
               
        with h5py.File(filename, 'r') as f:
            
            ones = np.ones((184,1))
            
            for i in range(0,num_of_positions):
                
                string = "{:02d}".format(i)
                                
                cPoints.append(
                    np.concatenate((f["camera/points"][string][I,:].astype(np.float64), ones[I]), axis=1))
                pPoints.append(
                    np.concatenate((f["projector/points"][string][np.invert(I),:].astype(np.float64), ones[np.invert(I)]), axis=1))
                
        self.setProjectorPoints(pPoints)
        self.setCameraPoints(cPoints)
        
    def setCameraBoardPoints(self,dataIn):
        
        assert dataIn.shape[1]<dataIn.shape[0]
        self.cBoard = tf.constant(dataIn, dtype = DATATYPE)
        
    def setProjectorBoardPoints(self,dataIn):
        
        assert dataIn.shape[1]<dataIn.shape[0]
        self.pBoard = tf.constant(dataIn, dtype = DATATYPE)

    def getInput1D(self):
        cOut = self.cPoints.concat()
        pOut = self.pPoints.concat()
                
        return tf.transpose(self.cBoard),tf.transpose(self.pBoard),tf.reshape(cOut, (-1,1)),tf.reshape(pOut, (-1,1))
    
    def getInitParams(self):
        return self.parameterVector
    
    def loadEstimateFromModelA(self, filename, n=0):
        
        nI = self.nI
        
        with h5py.File(filename, 'r') as f:
            
            camArray = f[r"/camera/array"][()]
            projArray = f[r"/projector/array"][()]
            extArray = f[r"/extrinsic/array"][()]
            
        self.parameterVector = np.concatenate((
            camArray[:nI].reshape(-1,1),
            projArray[:nI].reshape(-1,1),
            extArray.reshape(-1,1),
            camArray[nI:].reshape(-1,1)))
        
        self.parameterVector =  tf.Variable(self.parameterVector.flatten(), dtype = DATATYPE)
            

            
        
        