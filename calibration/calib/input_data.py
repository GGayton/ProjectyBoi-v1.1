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
            
            #Load all poses
            self.pose_ids = self.points[self.keys[0]].keys()

            if len(self.keys)>1:
                for key in self.keys[1:]:
                    self.pose_ids = list(set(self.pose_ids) | set(self.points[key].keys()))
            self.pose_ids = sorted(self.pose_ids)
    
    def get_inputs_TF(self):
        out = {}
        art = {}
        for key in self.keys:
            point_set = []
            art_set = []

            #Construct a ragged tensor for each component. If a particular pose is not
            #used for that component, that slice of the ragged tensor is blank
            for j in range(self.num_positions):

                pose_id = "{:02d}".format(j)
                

                if pose_id in list(self.points[key].keys()):
                    point_set += [self.points[key][pose_id].reshape(-1,1)]
                    art_set += [self.artefact[key][pose_id]]
                else:
                    point_set += []
                    art_set += []

            out[key] = tf.constant(np.vstack(point_set), dtype = self.datatype)
            art[key] = tf.ragged.constant(art_set, dtype=self.datatype)

        return out,art

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

    def load_estimate(self, filename):
        
        #Obtain keys
        with h5py.File(filename, 'r') as f:
            temp_keys = list(f.keys())

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

                self.params[key]["extrinsic"] = {}
                self.params[key]["extrinsic"]["rotation"] = {}
                self.params[key]["extrinsic"]["translation"] = {}

                for pose_id in f[key]["extrinsic"].keys():

                    self.params[key]["extrinsic"]["rotation"][pose_id] =\
                        f[key]["extrinsic"][pose_id]["rotation"][()]
                    self.params[key]["extrinsic"]["translation"][pose_id] =\
                        f[key]["extrinsic"][pose_id]["translation"][()]

    def get_serial_estimate(self):
        params = {}
        num_poses = len(self.pose_ids)
        for key in self.keys:

            num_dist_params = self.params[key]["distortion"].shape[0]
            temp = np.zeros(5+num_dist_params)
            temp[:5] = self.params[key]["matrix"][[0,1,0,0,1],[0,1,1,2,2]]
            temp[5:5+num_dist_params] = self.params[key]["distortion"]

            for id in self.pose_ids:
                if id in self.params[key]["extrinsic"]["rotation"].keys():
                    temp = np.concatenate((
                        temp,
                        self.params[key]["extrinsic"]["rotation"][id].flatten(),
                        self.params[key]["extrinsic"]["translation"][id].flatten()
                    ),axis=0)
                else:
                    temp = np.concatenate((
                        temp,
                        np.zeros(6)
                    ),axis=0)
                    
            params[key] = tf.constant(temp,dtype = self.datatype)

        return params

    @staticmethod
    def check_list(list1,list2):
        return sorted(list1) == sorted(list2)

    def checkstore_keys(self,temp_keys):
        if self.keys==None:
            self.keys = temp_keys
        else:
            assert self.check_list(temp_keys, self.keys), \
            "There is a mismatch in the keys already realised - to reset, set self.keys=None."
            
   