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
    
    def __init__(self, num_of_points, num_of_positions):
        super().__init__()
        self.nK = 5
        self.nD = 7
        self.nI = self.nK + self.nD
                
        self.optimise = tf.function(self.train, 
                                input_signature=(tf.TensorSpec(shape=[3,num_of_points], dtype=DATATYPE),
                                                tf.TensorSpec(shape=[num_of_points*num_of_positions*2,1], dtype=DATATYPE),
                                                tf.TensorSpec(shape=[self.nI+num_of_positions*6], dtype=DATATYPE),
                                                tf.TensorSpec(shape=[], dtype=DATATYPE),
                                                tf.TensorSpec(shape=[], dtype=tf.int32),
                                                tf.TensorSpec(shape=[], dtype=tf.int32),
                                                tf.TensorSpec(shape=[], dtype=DATATYPE),
                                                tf.TensorSpec(shape=[], dtype=tf.int32),
                                                ))
               
        self.weightedOptimise = tf.function(self.weightedTrain, 
                input_signature=(tf.TensorSpec(shape=[3,num_of_points], dtype=DATATYPE),
                                tf.TensorSpec(shape=[num_of_points*num_of_positions*2,1], dtype=DATATYPE),
                                tf.TensorSpec(shape=[self.nI+num_of_positions*6], dtype=DATATYPE),
                                tf.TensorSpec(shape=[num_of_points*num_of_positions*2,num_of_points*num_of_positions*2], dtype=DATATYPE),
                                tf.TensorSpec(shape=[], dtype=DATATYPE),
                                tf.TensorSpec(shape=[], dtype=tf.int32),
                                tf.TensorSpec(shape=[], dtype=tf.int32),
                                tf.TensorSpec(shape=[], dtype=DATATYPE),
                                tf.TensorSpec(shape=[], dtype=tf.int32),
                                ))
    
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
    def transformFunction(self,x,X):
        
        print("Tracing: transformFunction")
        
        nK = self.nK
        nI = self.nI
        
        k = X[:nK]
        D = X[nK:nI]
        e_all = X[nI:]
        
        K = self.cameraMatrix(k)
                
        extrinsics = tf.split(e_all, e_all.shape[0]//6)   
        
        #obtain grad
        r,t = tf.split(extrinsics[0], [3,3])
        R = self.rodrigues(r)
        T = tf.reshape(t, (3,1)) 
        
        allEst = R @ x + T
        
        for i in range(1,len(extrinsics)):
                                           
            #obtain grad
            r,t = tf.split(extrinsics[i], [3,3])
        
            R = self.rodrigues(r)
    
            T = tf.reshape(t, (3,1)) 
            
            estimate = R @ x + T
            
            allEst = tf.concat((allEst, estimate),axis=1)
        
        allEst = allEst/allEst[2,:]
        
        allEst = self.distort(allEst, D)
        
        allEst = K @ allEst
        
        allEst = tf.reshape(tf.transpose(allEst[:2,:]), (-1,1))
        
        return allEst 
                
    @tf.function(input_signature=(tf.TensorSpec(shape=[3,None], dtype=DATATYPE),
                              tf.TensorSpec(shape=[None], dtype=DATATYPE),
                              ))
    def gradFunction(self,x,X):
        print("Tracing: gradFunction")
        
        nK = self.nK
        nI = self.nI
        
        with tf.GradientTape(persistent = False) as tape:
            
            #Set up inputs
            k = X[:nK]
            D = X[nK:nI]
            e = X[nI:]
            
            tape.watch(k)
            tape.watch(D)
            tape.watch(e)
            
            #obtain extrinsics
            r,t = tf.split(e, [3,3])
            R = self.rodrigues(r)
            T = tf.reshape(t, (3,1)) 
            
            #obtain intrinsics       
            K = self.cameraMatrix(k)
                                        
            estimate = self.backProject(x,K,R,T,D)
            estimate = tf.reshape(tf.transpose(estimate), (-1,1))[:,0]
                
        grad = tape.jacobian(estimate, [k,D,e])
                
        return grad
    
    @tf.function
    def jacobianFunction(self,x,X):
        
        print("Tracing: jacobianFunction")
        nK = self.nK
        nD = self.nD
        nI = self.nI

        intrinsics = X[:nI]
        extrinsicsAll = X[nI:]
        
        numPositions = extrinsicsAll.shape[0]//6
        numPoints = x.shape[1]//1
    
        extrinsics = tf.split(extrinsicsAll, numPositions)   
        
        jacobian = tf.zeros((numPoints * numPositions * 2, numPositions*6 + nI),dtype=DATATYPE)
        
        #create base indices for scattering updates into jacobian with
        i,j = tf.meshgrid(tf.linspace(0,numPoints*2-1,numPoints*2),tf.linspace(0,nK-1,nK), indexing='ij')
        i = tf.cast(tf.reshape(i, (numPoints*nK*2,1)), dtype = tf.int32)
        j = tf.cast(tf.reshape(j, (numPoints*nK*2,1)), dtype = tf.int32)
        base_k_indices = tf.concat((i,j),axis=1)
        
        i,j = tf.meshgrid(tf.linspace(0,numPoints*2-1,numPoints*2),tf.linspace(nK,nI-1,nD), indexing='ij')
        i = tf.cast(tf.reshape(i, (numPoints*nD*2,1)), dtype = tf.int32)
        j = tf.cast(tf.reshape(j, (numPoints*nD*2,1)), dtype = tf.int32)
        base_d_indices = tf.concat((i,j),axis=1)
        
        i,j = tf.meshgrid(tf.linspace(0,numPoints*2-1,numPoints*2),tf.linspace(nI,nI+5,6), indexing='ij')
        i = tf.cast(tf.reshape(i, (numPoints*6*2,1)), dtype = tf.int32)
        j = tf.cast(tf.reshape(j, (numPoints*6*2,1)), dtype = tf.int32)
        base_e_indices = tf.concat((i,j),axis=1)
        
        i=0
        for extrinsic in extrinsics:
            
            #parameter set for each position
            parameters = tf.concat((intrinsics,extrinsic),axis=0)
            
            #constants to be added for each iteration
            i_start = i*numPoints*2
            r_start = i*6
            
            #obtain grad
            grad = self.gradFunction(x,parameters)

            #scatter in k values
            kGrad = tf.reshape(grad[0], (numPoints*nK*2,1))[:,0]
            jacobian = tf.tensor_scatter_nd_update(
                jacobian, 
                base_k_indices + tf.constant((i_start,0), dtype=tf.int32), 
                kGrad
                )
            
            #scatter in D values
            DGrad = tf.reshape(grad[1], (numPoints*nD*2,1))[:,0]
            jacobian = tf.tensor_scatter_nd_update(
                jacobian, 
                base_d_indices + tf.constant((i_start,0), dtype=tf.int32), 
                DGrad
                )
            
            #scatter in extrinsics
            eGrad = tf.reshape(grad[2], (numPoints*6*2,1))[:,0]
            jacobian = tf.tensor_scatter_nd_update(
                jacobian, 
                base_e_indices + tf.constant((i_start,r_start), dtype=tf.int32), 
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
        tf.TensorSpec(shape=[None], dtype=DATATYPE),
        ))
    def singleHamiltonianFunction(self,x,X):
        
        print("Tracing: singleHamiltonianFunction")
        nK = self.nK
        nI = self.nI
                
        with tf.GradientTape(persistent = True) as tape1:
            tape1.watch(X)
            with tf.GradientTape(persistent = True) as tape2:
                
                tape2.watch(X)
                
                #Set up inputs
                k = X[:nK]
                D = X[nK:nI]
                e = X[nI:]
                        
                #obtain extrinsics
                r,t = tf.split(e, [3,3])
                R = self.rodrigues(r)
                T = tf.reshape(t, (3,1)) 
                
                #obtain intrinsics       
                K = self.cameraMatrix(k)
                                
                y = self.backProject(x[:,0:1], K, R, T, D)
                
                y1 = y[0,0]
                y2 = y[1,0]
            
            dy1dx = tape2.gradient(y1, X)
            dy2dx = tape2.gradient(y2, X)

        d2y1dx2 = tape1.jacobian(dy1dx,X)
        d2y2dx2 = tape1.jacobian(dy2dx,X)

        return d2y1dx2,d2y2dx2
    
    @tf.function()
    def blockHamiltonianFunction(self,x,X):
        
        print("Tracing: blockHamiltonianFunction")
        N = x.shape[1]

        H=tf.TensorArray(tf.float64, size=N*2)
        
        for i in tf.range(x.shape[1]):
            
            d2y1dx2,d2y2dx2=self.singleHamiltonianFunction(x[:,i:i+1], X)
            
            H = H.write(2*i, d2y1dx2)
            H = H.write(2*i+1, d2y2dx2)

        return H.stack()    
                
    def fullHamiltonianFunction(self,x,X):
    

        nI = self.nI
        
        N = x.shape[1]
        
        intrinsics = X[:nI]
        extrinsics = X[nI:]
        
        numPositions = extrinsics.shape[0]//6
        
        H = np.zeros((nI+6, nI+6, N*2*numPositions))
        
        extrinsics = tf.split(extrinsics, numPositions)
        bar = ProgressBar()
        bar.updateBar(0,numPositions)
        
        for j in range(numPositions):
            
            #parameter set for each position
            parameters = tf.concat((intrinsics,extrinsics[j]),axis=0)
                        
            Htemp = self.blockHamiltonianFunction(x, parameters)
            
            Htemp = Htemp.numpy()
            
            Htemp = np.swapaxes(Htemp, 0,2)
            
            H[:,:,j*N*2:(j+1)*N*2] = Htemp
            
            bar.updateBar(j+1, numPositions)
        return H
    
    def boardGradFunction(self,x,X):        
        nK = self.nK
        nI = self.nI
        
        with tf.GradientTape(persistent = False) as tape:
            
            #Set up inputs
            k = X[:nK]
            D = X[nK:nI]
            e = X[nI:]
            
            tape.watch(x)
            
            #obtain extrinsics
            r,t = tf.split(e, [3,3])
            R = self.rodrigues(r)
            T = tf.reshape(t, (3,1)) 
            
            #obtain intrinsics       
            K = self.cameraMatrix(k)
                                        
            estimate = self.backProject(x,K,R,T,D)
            estimate = tf.reshape(tf.transpose(estimate), (-1,1))[:,0]
                            
        grad = tape.jacobian(estimate, x)
        
        return grad
    
    @tf.function
    def boardJacobianFunction(self,x,X):
        
        print("Tracing: jacobianFunction")
        nK = self.nK
        nD = self.nD
        nI = self.nI

        intrinsics = X[:nI]
        extrinsicsAll = X[nI:]
        
        numPositions = extrinsicsAll.shape[0]//6
        numPoints = x.shape[1]//1
    
        extrinsics = tf.split(extrinsicsAll, numPositions)   
        
        jacobian = tf.zeros((numPoints * numPositions * 2, numPoints*3),dtype=DATATYPE)
        
        #create base indices for scattering updates into jacobian with
        i,j = tf.meshgrid(tf.linspace(0,numPoints*2-1,numPoints*2),tf.linspace(0,numPoints-1,numPoints)*3, indexing='ij')
        i = tf.cast(tf.reshape(i, (2*numPoints**2,1)), dtype = tf.int32)
        j = tf.cast(tf.reshape(j, (2*numPoints**2,1)), dtype = tf.int32)
        baseIndices = tf.concat((i,j),axis=1)
                
        i=0
        for extrinsic in extrinsics:
            
            #parameter set for each position
            parameters = tf.concat((intrinsics,extrinsic),axis=0)
            
            #constants to be added for each iteration
            i_start = i*numPoints*2
            
            #obtain grad
            grad = self.boardGradFunction(x,parameters)

            #scatter in k values
            xGrad = tf.reshape(grad[:,0,:], (2*numPoints**2,1))[:,0]
            jacobian = tf.tensor_scatter_nd_update(
                jacobian, 
                baseIndices + tf.constant((i_start,0), dtype=tf.int32), 
                xGrad
                )
                  
            yGrad = tf.reshape(grad[:,1,:], (2*numPoints**2,1))[:,0]
            jacobian = tf.tensor_scatter_nd_update(
                jacobian, 
                baseIndices + tf.constant((i_start,1), dtype=tf.int32), 
                yGrad
                )
            
            zGrad = tf.reshape(grad[:,2,:], (2*numPoints**2,1))[:,0]
            jacobian = tf.tensor_scatter_nd_update(
                jacobian, 
                baseIndices + tf.constant((i_start,2), dtype=tf.int32), 
                zGrad
                )
            i=i+1
    
        return jacobian
                                   
#%% InputData
class InputData(DataTF):
    
    def __init__(self):
        self.nD = 7
        self.nI = 5+self.nD
    
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
                    np.concatenate((f["camera/points"][string][()][I,:].astype(np.float64), ones[I]), axis=1))
                pPoints.append(
                    np.concatenate((f["projector/points"][string][()][np.invert(I),:].astype(np.float64), ones[np.invert(I)]), axis=1))
                
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








