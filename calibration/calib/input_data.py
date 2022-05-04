import numpy as np
import tensorflow as tf
import h5py
from commonlib.h5py_functions import num_of_keys

class InputData():
    
    def __init__(self, datatype=tf.float64):
        self.datatype = datatype

    def load_inputs(self, filename):
        self.num_positions = num_of_keys(filename, "inputs")
        with h5py.File(filename, 'r') as f:
            
            #Identify all components by their second level key
            self.keys = list(f["inputs"]["00"].keys())
            for key in f["inputs"].keys():
                for ckey in f["inputs"][key].keys():
                    if ckey not in self.keys: self.keys.append(ckey)

            #initialise the point dictionary
            self.points = {}
            self.artefact = {}
            self.keymap = np.empty((self.num_positions, len(self.keys)), dtype = int)
            tkey = list(f["inputs"].keys())
            for i in range(len(self.keys)):
                self.points[self.keys[i]] = {}
                self.artefact[self.keys[i]] = {}
                
                for j in range(len(tkey)):
                    if self.keys[i] in list(f["inputs"][tkey[j]].keys()):
                        self.points[self.keys[i]][tkey[j]] = f["inputs"][tkey[j]][self.keys[i]]["points"][()]
                        self.artefact[self.keys[i]][tkey[j]] = f["inputs"][tkey[j]][self.keys[i]]["artefact"][()]
                        self.keymap[j,i] = self.points[self.keys[i]][tkey[j]].shape[0]

                        assert self.points[self.keys[i]][tkey[j]].shape[0] == self.artefact[self.keys[i]][tkey[j]].shape[0],\
                            "Number of measured points and artefact points must be the same."
                    else:
                        self.points[self.keys[i]][tkey[j]] = np.nan
                        self.artefact[self.keys[i]][tkey[j]] = np.nan
                        self.keymap[j,i] = 0

    def get_inputs_TF(self):
        out = [[] for _ in range(len(self.keys))]
        for i in range(len(self.keys)):
            temp = []
            for j in range(self.num_positions):
                temp += [self.points[self.keys[i]]["{:02d}".format(j)].T.reshape(-1,1)]
            out[i] = np.vstack(temp)
        
        return out

    def get_inputs_numpy(self):
        out = [[] for _ in range(len(self.keys))]
        art = [[] for _ in range(len(self.keys))]

        for i in range(len(self.keys)):
            point_set = []
            art_set = []
            for j in range(self.num_positions):
                point_set += [self.points[self.keys[i]]["{:02d}".format(j)]]
                art_set += [self.artefact[self.keys[i]]["{:02d}".format(j)]]

            out[i] = point_set
            art[i] = art_set
        
        return out,art

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


            
    def getInput(self):
        cOut = tf.transpose(self.cPoints.concat())
        pOut = tf.transpose(self.pPoints.concat())
                
        return tf.transpose(self.board),cOut,pOut
    
    def getInput1D(self):
        cOut = self.cPoints.concat()
        pOut = self.pPoints.concat()
                
        return tf.transpose(self.board),tf.reshape(cOut, (-1,1)),tf.reshape(pOut, (-1,1))