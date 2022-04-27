import numpy as np
import tensorflow as tf
import h5py
from commonlib.h5py_functions import num_of_keys

DATATYPE = tf.float64
   
class DataTF():
            
    def setCameraPoints(self,dataIn):
        self.cPoints = dataIn.copy()
        
        for i in range(len(self.cPoints)):
            assert self.cPoints[i].shape[1]<self.cPoints[i].shape[0]
            self.cPoints[i] = self.cPoints[i][:,:2]
            
        self.cPoints = self.moveToGPU(self.cPoints)
        
        self.positionNum = len(dataIn)
        
    def setProjectorPoints(self,dataIn):
        self.pPoints = dataIn.copy()
        
        for i in range(len(self.pPoints)):
            assert self.pPoints[i].shape[1]<self.pPoints[i].shape[0]
            self.pPoints[i] = self.pPoints[i][:,:2]
            
        self.pPoints = self.moveToGPU(self.pPoints)
        
    def setBoardPoints(self,dataIn):
        
        self.board = dataIn
        # self.board[:,2] = 0
        assert self.board.shape[1]<self.board.shape[0]
        self.board = tf.constant(self.board, dtype = DATATYPE)
        
        self.pointNum = self.board.shape[0]
        
    def loadEstimate(self, filename, nDist):
        
        with h5py.File(filename, 'r') as f:
            
            camArray = f[r"/camera/array"][()]
            projArray = f[r"/projector/array"][()]
            extArray = f[r"/extrinsic/array"][()]
            
        camArray = np.concatenate((camArray[:5], np.ones(nDist)*0.1, camArray[5:]))
        projArray = np.concatenate((projArray[:5], np.ones(nDist)*0.1, projArray[5:]))
            
        self.cameraParameterVector = tf.Variable(camArray.flatten(), dtype = DATATYPE)
        self.projectorParameterVector = tf.Variable(projArray.flatten(), dtype = DATATYPE)
        self.extrinsicParameterVector = tf.Variable(extArray.flatten(), dtype = DATATYPE)
                
    def loadParametersFromSeed(self,filename):
        
        with h5py.File(filename, 'r') as f:
            
            camArray = f[r"/camera/array"][()]
            projArray = f[r"/projector/array"][()]
            extArray = f[r"/extrinsic/array"][()]
            
        self.cameraParameterVector = tf.Variable(camArray.flatten(), dtype = DATATYPE)
        self.projectorParameterVector = tf.Variable(projArray.flatten(), dtype = DATATYPE)
        self.extrinsicParameterVector = tf.Variable(extArray.flatten(), dtype = DATATYPE)
        
    def loadBoardPoints(self, filename):
        
        with h5py.File(filename, 'r') as f:
            
            board = f["board"][:,:]
            
        f.close()
        
        if board.shape[1] == 2:
            board = np.concatenate((board, np.zeros_like(board[:,0:1])), axis=1)
        board = board.astype(np.float64)
            
        self.setBoardPoints(board)
        
    def loadMeasuredPoints(self, filename):
                
        num_of_positions = num_of_keys(filename, "camera/points")
        
        cPoints = []
        pPoints = []
                
        with h5py.File(filename, 'r') as f:
            
            ones = np.ones((184,1))
            
            for i in range(0,num_of_positions):
                
                string = "{:02d}".format(i)
                                
                cPoints.append(
                    np.concatenate((f["camera/points"][string][:,:].astype(np.float64), ones), axis=1))
                pPoints.append(
                    np.concatenate((f["projector/points"][string][:,:].astype(np.float64), ones), axis=1))
        
        f.close()
    
        self.setProjectorPoints(pPoints)
        self.setCameraPoints(cPoints)

    def moveToGPU(self,arrayList):
        
        numArrays = len(arrayList)
        
        tArray = tf.TensorArray(
            dtype = DATATYPE, 
            size = numArrays, 
            dynamic_size = False,
            clear_after_read = False)
        
        for i in range(numArrays):
            
            tArray = tArray.write(i, tf.constant(arrayList[i], dtype = DATATYPE))
        
        return tArray
    
    def getInitCamParams(self):
        return self.cameraParameterVector
    
    def getInitProjParams(self):
        return self.projectorParameterVector
    
    def getInitExtParams(self):
        return self.extrinsicParameterVector
    
    def getInitParams(self):
        return self.cameraParameterVector,self.projectorParameterVector,self.extrinsicParameterVector
    
    def getInput(self):
        cOut = tf.transpose(self.cPoints.concat())
        pOut = tf.transpose(self.pPoints.concat())
                
        return tf.transpose(self.board),cOut,pOut
    
    def getInput1D(self):
        cOut = self.cPoints.concat()
        pOut = self.pPoints.concat()
                
        return tf.transpose(self.board),tf.reshape(cOut, (-1,1)),tf.reshape(pOut, (-1,1))