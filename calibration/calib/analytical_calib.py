import numpy as np
from commonlib.rotations import rodrigues, rodriguesInv, approximate_matrix_as_R

class AnalyticalCalibration():

    def normalisation_matrix(self,points):
        
        meanX = np.mean(points[:,0])
        meanY = np.mean(points[:,1])
        
        mean_distX = np.mean( np.abs(points[:,0] - meanX) )
        mean_distY = np.mean( np.abs(points[:,1] - meanY) )
        
        A = np.array([
            [1/mean_distX,0,0],
            [0,1/mean_distY,0],
            [0,0,1]])
        
        B = np.array([
            [1,0,-meanX],
            [0,1,-meanY],
            [0,0,1]])
        
        T = A@B
        
        return T 

    def solveSVD(self,A):
            
        u, s, vh = np.linalg.svd(A)
        null_space = np.compress(s == s.min(), vh, axis=0)
        return null_space.T
        
    def homography(self,objectPoints,cameraPoints):
        
        N = objectPoints.shape[0]
        
        O = objectPoints
        
        To = self.normalisation_matrix(objectPoints)

        O = O @ To.T
            
        ones = np.ones((N,1))
        zeros = np.zeros((N,1))
        
        P = cameraPoints
        
        Tp = self.normalisation_matrix(cameraPoints)
        Tp_inv = np.linalg.inv(Tp)
        
        P = P @ Tp.T
        
        b = np.empty((2*N, 9))
        
        even = np.concatenate(
            [-O[:,0].reshape(-1,1),
             -O[:,1].reshape(-1,1),
             -ones,
             zeros,
             zeros,
             zeros,
             (P[:,0] * O[:,0]).reshape(-1,1),
             (P[:,0] * O[:,1]).reshape(-1,1),
             P[:,0].reshape(-1,1)], axis=1
            )
        
        odd = np.concatenate(
            [ 
            zeros, 
            zeros, 
            zeros,
            -O[:,0].reshape(-1,1),
            -O[:,1].reshape(-1,1),
            -ones,
            (P[:,1] * O[:,0]).reshape(-1,1),
            (P[:,1] * O[:,1]).reshape(-1,1),
            P[:,1].reshape(-1,1)], axis=1
            )
        
        for i in range(N):
            b[2*i,:] = even[i,:]
            b[2*i+1,:] = odd[i,:]
                        
        h = self.solveSVD(b).reshape(-1)
        
        homography = np.array([
            [h[0], h[1], h[2]],
            [h[3], h[4], h[5]],
            [h[6], h[7], h[8]]
            ])
        
        H = Tp_inv @ homography @ To
        
        H = H/H[2,2]
                
        return H

    def assemble_V(self,H,i,j):
        
        H = H.T
        
        vij = np.array([
        [H[i,0] * H[j,0]],
        [H[i,0] * H[j,1] + H[i,1] * H[j,0]],
        [H[i,1] * H[j,1]],
        [H[i,2] * H[j,0] + H[i,0] * H[j,2]],
        [H[i,2] * H[j,1] + H[i,1] * H[j,2]],
        [H[i,2] * H[j,2]]
        ])
        
        return vij

    def get_parameters(self,H,inv_fx=False,inv_fy=False):
            
        positionsNum = len(H)
    
        #Estimate intrinsic parameters
        V = np.empty((2*positionsNum, 6))
        for i in range(0,positionsNum):
            
            v11 = self.assemble_V(H[i],0,0)
            v12 = self.assemble_V(H[i],0,1)
            v22 = self.assemble_V(H[i],1,1)
            
            V[2*i,:] = v12.T
            V[2*i+1,:] = (v11 - v22).T
        
        b = self.solveSVD(V)
            
        B11 = b[0]
        B12 = b[1]
        B22 = b[2]
        B13 = b[3]
        B23 = b[4]
        B33 = b[5]
        
        v0 = (B12 * B13 - B11 * B23)/(B11 * B22 - B12**2)
        scale = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23))/B11
        
        if inv_fx == False:
            fx = (scale/B11)**0.5
        elif inv_fx == True:
            fx = -(scale/B11)**0.5
            
        if inv_fy == False:
            fy = (scale * B11 / (B11 * B22 - B12**2))**0.5
        elif inv_fy == True:
            fy = -(scale * B11 / (B11 * B22 - B12**2))**0.5
            
        u0 = - B13 * fx**2/scale
        s = -B12 * fx**2 * fy / scale 
        A = np.array([
            [fx[0], s[0], u0[0]],
            [0, fy[0], v0[0]],
            [0,0,1]
            ])
        
        A_inv = np.linalg.inv(A)
        
        tList = np.empty((3,positionsNum))
        rList = np.empty((3,positionsNum))
        for i in range(positionsNum):
            r1 = A_inv @ H[i][:,0:1]
            r2 = A_inv @ H[i][:,1:2]
            
            _lambda = (np.sum(r1**2))**0.5
            
            r1 = r1/_lambda
            r2 = r2/_lambda
            r3 = np.cross(r1, r2,0,0,0)
            
            Q = np.concatenate((r1, r2, r3), axis=1)
            
            R = approximate_matrix_as_R(Q)
            
            t = A_inv @ H[i][:,2]/_lambda
            
            tList[:,i] = t
            rList[:,i] = rodriguesInv(R)     
            
        return A, tList, rList
    
    def calibrate(self,object_points,camera_points,inv_fx=False,inv_fy=False):
        
        assert len(object_points) == len(camera_points), "Missing camera or object points."

        #Estimate Homography
        H = []
        for i in range(len(camera_points)):
            H.append(self.homography(object_points[i], camera_points[i]))
            
        #Get parameters
        return self.get_parameters(H, inv_fx = inv_fx, inv_fy = inv_fy) 
    
    def estimate_extrinsics(self,r,t):

        num_cameras = len(r)
        R = [[] for _ in range(num_cameras)]
        T = [[] for _ in range(num_cameras)]
        R[0] = np.array([0,0,0])
        T[0] = np.array([0,0,0])

        for i in range(1,num_cameras):

            R[i],T[i] = self.estimate_extrinsic(r[0],t[0],r[i],t[i])

        return R,T

    def estimate_extrinsic(self,rc,tc,rp,tp):
        
        positionsNum = tc.shape[1]
        
        r_out = np.empty((3,positionsNum))
        t_out = np.empty((3,positionsNum))
        
        for i in range(tc.shape[1]):
            
            Rc = rodrigues(rc[:,i:i+1])
            Rp = rodrigues(rp[:,i:i+1])
            
            r_out[:,i] = rodriguesInv(Rp @ Rc.T)
            
            t_out[:,i:i+1] = tp[:,i:i+1] - Rp@Rc.T@tc[:,i:i+1]
                
        r_out = np.mean(r_out, axis=1)
        t_out = np.mean(t_out, axis = 1)
        
        return r_out,t_out
            
        
        