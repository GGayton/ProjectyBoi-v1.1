import h5py
from commonlib.rotations import rodrigues

class InputParameter():  
       
    def __init__(self):
        pass

    def load_parameters(self,filename):
                
        with h5py.File(filename, 'r') as f:

            keys = list(f.keys())
            assert "camera" in keys, "Can't find camera parameters"
            assert "projector" in keys, "Can't find projector parameters"

            Kc = f["camera"]["matrix"][()]
            Dc = f["camera"]["distortion"][()]

            Kp = f["projector"]["matrix"][()]
            Dp = f["projector"]["distortion"][()]

            r = f["projector"]["rotation"][()]
            t = f["projector"]["translation"][()]
                
    def get(self):
        R = rodrigues(self.r[0], self.r[1], self.r[2])
        return self.Kc,self.Dc,self.Kp,self.Dp,R,self.t
    


        
        