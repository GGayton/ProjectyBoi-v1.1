import numpy as np
import tensorflow as tf
import h5py
from commonlib.h5py_functions import num_of_keys

class InputData():
    
    def __init__(self, datatype=tf.float64):
        self.datatype = datatype
        self.keys = None

    def load_inputs(self, filename):
        self.num_positions = num_of_keys(filename, "inputs")
        with h5py.File(filename, 'r') as f:
            
            #Identify all components by their second level key
            temp_keys = list(f["inputs"]["00"].keys())
            for key in f["inputs"].keys():
                for ckey in f["inputs"][key].keys():
                    if ckey not in temp_keys: temp_keys.append(ckey)

            #Either store keys, or check they are consistent
            self.checkstore_keys(temp_keys)

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
    
    def get_inputs_dict(self):
        return self.points,self.artefact

    def get_pose_IDs(self):

        assert self.keys!=None, "You must load the inputs first."

        out = {}
        for key in self.keys:
            out[key] = list(self.points[key].keys())

        return out

    def load_estimate(self, filename):
        
        #Obtain keys
        with h5py.File(filename, 'r') as f:
            temp_keys = list(f.keys)

        #Either store keys, or check they are consistent
        self.checkstore_keys(temp_keys)

        #Initialise parameter dict
        self.params = {}
        for key in self.keys:
            self.params[key] = {}

        #Load all parameters
        with h5py.File(filename, 'r') as f:
            for key in self.keys:
                self.params[key]["matrix"] = f[key]["matrix"][()]
                self.params[key]["distortion"] = f[key]["distortion"][()]
                self.params[key]["rotation"] = f[key]["rotation"][()]
                self.params[key]["translation"] = f[key]["translation"][()]

                self.params[key]["rotation"] = f[key]["rotation"][()]
                self.params[key]["rotation"] = f[key]["rotation"][()]




    @staticmethod
    def check_list(list1,list2):
        return sorted(list1) == sorted(list2)

    def checkstore_keys(self,temp_keys):
        if self.keys==None:
            self.keys = temp_keys
        else:
            assert self.check_list(temp_keys, self.keys), \
            "There is a mismatch in the keys already realised - to reset, set self.keys=None."



            
    def getInput(self):
        cOut = tf.transpose(self.cPoints.concat())
        pOut = tf.transpose(self.pPoints.concat())
                
        return tf.transpose(self.board),cOut,pOut
    
    def getInput1D(self):
        cOut = self.cPoints.concat()
        pOut = self.pPoints.concat()
                
        return tf.transpose(self.board),tf.reshape(cOut, (-1,1)),tf.reshape(pOut, (-1,1))