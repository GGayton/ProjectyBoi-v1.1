import tensorflow as tf
import numpy as np
import h5py


from calib.base_calib import Calibration
from commonlib.console_outputs import ProgressBar
from commonlib.h5py_functions import num_of_keys

#%% Calibration
class SerialCalibration(Calibration):

    def __init__(self, datatype = tf.float64):
        self.datatype = datatype

        #number of parameters for the camera matrix and projector matrix
        self.nK = 5
        self.nD = 7
        self.nI = self.nD + self.nK

        super().__init__(datatype)
        self.init_serial_functions()

    def init_serial_functions(self):
        print("Tracing... ", end = "")

        self.distort_TF = tf.function(self.distort, input_signature=(
            tf.TensorSpec(shape=[3,None], dtype=self.datatype),
            tf.TensorSpec(shape=[None], dtype=self.datatype)
            ))
        self.distort_TF = self.distort_TF.get_concrete_function()

        self.back_project_TF = tf.function(self.back_project, input_signature=(
            tf.TensorSpec(shape=[3,None], dtype=self.datatype),
            tf.TensorSpec(shape=[3,3], dtype=self.datatype),
            tf.TensorSpec(shape=[3,3], dtype=self.datatype),
            tf.TensorSpec(shape=[3,1], dtype=self.datatype),
            tf.TensorSpec(shape=[self.nD], dtype=self.datatype),
            ))
        self.back_project_TF = self.back_project_TF.get_concrete_function()

        self.transform_TF = tf.function(self.transform, input_signature=(
            tf.RaggedTensorSpec(shape=[3,None,None],dtype=self.datatype,ragged_rank=1),
            tf.TensorSpec(shape=[None], dtype=self.datatype)
            ))
        # self.transform_TF = self.transform_TF.get_concrete_function()

        self.gradient_TF = tf.function(self.gradient,input_signature=(
            tf.TensorSpec(shape=[3,None], dtype=self.datatype),
            tf.TensorSpec(shape=[None], dtype=self.datatype),
            ))

        self.jacobian_TF = tf.function(self.jacobian, input_signature=(
            tf.RaggedTensorSpec(shape=[3,None,None],dtype=self.datatype,ragged_rank=1),
            tf.TensorSpec(shape=[None], dtype=self.datatype)
        ))
        print("done")
    
    def distort(self,x,X):
                
        k1 = X[0]
        k2 = X[1]
        k3 = X[2]
        p1 = X[3]
        p2 = X[4]
        x0 = X[5]
        y0 = X[6]
        
        offset = tf.stack((x0,y0,tf.constant(1,self.datatype)), axis=0)
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

    def back_project(self,x,K,R,T,D):
                
        estimate = R @ x + T
        
        estimate = estimate/estimate[2,:]
        
        estimate = self.distort(estimate, D)
        
        estimate = K @ estimate
        
        return estimate[:2,:]
                           
    def transform(self,x,X):
        
        #Extract parameters from vector
        nK = self.nK
        nI = self.nI
        k = X[:nK]
        D = X[nK:nI]
        e_all = X[nI:]
        K = self.assemble_camera_matrix_TF(k)
        
        num_positions = x.shape[2]

        extrinsics = tf.split(e_all, num_positions)   
        
        #Transform first set
        r,t = tf.split(extrinsics[0], [3,3])
        R = self.rodrigues_TF(r)
        T = tf.reshape(t, (3,1))
        
        allEst = R @ x[:,:,0] + T
        
        #Transform next sets an ocncatenate onto first
        for i in range(1,len(extrinsics)):
                                           
            r,t = tf.split(extrinsics[i], [3,3])
            R = self.rodrigues_TF(r)
            T = tf.reshape(t, (3,1)) 
            
            estimate = R @ x[:,:,i] + T
            allEst = tf.concat((allEst, estimate),axis=1)
        
        #Proejctive transform and distort
        allEst = allEst/allEst[2,:]
        allEst = self.distort_TF(allEst, D)
        allEst = K @ allEst
        
        allEst = tf.reshape(tf.transpose(allEst[:2,:]), (-1,1))
        
        return allEst 
                
    def gradient(self,x,X):
        
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
    
    def jacobian(self,x,X):
        
        nK = self.nK
        nD = self.nD
        nI = self.nI

        intrinsics = X[:nI]
        extrinsicsAll = X[nI:]
        
        numPositions = extrinsicsAll.shape[0]//6
        numPoints = x.shape[1]//1
    
        extrinsics = tf.split(extrinsicsAll, numPositions)   
        
        jacobian = tf.zeros((numPoints * numPositions * 2, numPositions*6 + nI),dtype=self.datatype)
        
        #create base indices for scattering updates into jacobian with
        #(You can't do index assignment in TF, i.e. x[1,4] = 4)
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
             

