import tensorflow as tf

DATATYPE = tf.float64

class Calibration():
#%%   
    def __init__(self):
        return
    
    @tf.function(input_signature=(tf.TensorSpec(shape=[3], dtype=DATATYPE),))
    def rodrigues(self,r):
        
        print("Tracing: Rodrigues")
            
        theta = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(r)))
        
        zero = tf.zeros((1), dtype = DATATYPE)
        
        r_norm = r/theta
        
        r1,r2,r3 = tf.split(r_norm, 3)
        
        K = tf.concat([
            tf.concat([zero, tf.negative(r3), r2], 0),
            tf.concat([r3, zero, tf.negative(r1)], 0),
            tf.concat([tf.negative(r2), r1, zero], 0),
            ],0)
        
        K = tf.reshape(K, (3,3))
    
        R = tf.eye(3, dtype = DATATYPE) + tf.sin(theta)*K + (1-tf.cos(theta))*K@K
        
        return R
    
    @tf.function(input_signature=(tf.TensorSpec(shape=[3,3], dtype=DATATYPE),))
    def rodriguesInv(self,R):
        
        angle = tf.math.acos(( R[0,0] + R[1,1] + R[2,2] - 1)/2)
        
        denom = ((R[2,1] - R[1,2])**2 + (R[0,2] - R[2,0])**2 + (R[1,0] - R[0,1])**2)**0.5
        
        x = (R[2,1] - R[1,2]) / denom
        y = (R[0,2] - R[2,0]) / denom
        z = (R[1,0] - R[0,1]) / denom
        
        r = tf.concat(
            (
            tf.reshape(x,(1,1)),
            tf.reshape(y,(1,1)),
            tf.reshape(z,(1,1))
            ),
            axis = 0)
        
        r = r*angle
        
        return r
        
    @tf.function(input_signature=(tf.TensorSpec(shape=[5], dtype=DATATYPE),))
    def cameraMatrix(self,k):
            
        print("Tracing: Camera matrix")
        
        fx, fy, s, u0, v0 = tf.split(k,5)
        
        zero = tf.zeros((1), dtype = DATATYPE)
        one = tf.ones((1), dtype = DATATYPE)
        
        K = tf.concat([
            tf.concat([fx, s, u0], 0),
            tf.concat([zero, fy, v0], 0),
            tf.concat([zero, zero, one], 0),
            ],0)
        
        K = tf.reshape(K, (3,3))
        
        return K
                         
    #Approximate a rotation matrix from generic matrix
    def approximateMatrixWithR(self,Q):
        
        s,u,v = tf.linalg.svd(Q, full_matrices=True)
        R = u@tf.transpose(v)
        return R
               
    def printUpdate(self, Epoch, damping_factor, loss_sum):

        tf.print(
            "Epoch: ",Epoch,
            ", lambda: ", damping_factor,
            ", loss: ", loss_sum)
    
    def iterateDamping(self,dampingFactor, change):
        
        dampingFactor = dampingFactor*change
        
        if dampingFactor<1e-5: dampingFactor = dampingFactor*0 + 1e-5
        
        return dampingFactor
#%%          
    def train(self,x,y,X,dampingFactor,DISPLAY,FAILURE_COUNT_MAX,CHANGE_MIN,ITERATION_MAX):
        
        print("Tracing: Train")                
        optimised = tf.constant(False, dtype = tf.bool)
        
        failureCount = 0
                 
        epoch = tf.constant(0, dtype = tf.int32)
        
        #Initialise first loss and jacobian
        loss = y - self.transformFunction(x,X)
        lossSum = tf.math.reduce_sum(loss**2)
        J = self.jacobianFunction(x,X)
        
        lossSumChange = tf.constant(1,dtype = DATATYPE)
                
        while not optimised:
            
            JtJ = self.JtJFunction(J,dampingFactor)

            # update = tf.linalg.lstsq(JtJ, tf.transpose(J) @ loss, fast=False)
            update = tf.linalg.inv(JtJ) @ tf.transpose(J) @ loss
            
            # dir_div = self.geodesicAcceleration(x,X,update,J)
            
            # geoUpdate = tf.linalg.lstsq(JtJ, tf.transpose(J) @ dir_div, fast=False)/2
            
            # geoCondition = tf.math.less_equal(
            #     tf.reduce_sum(geoUpdate**2)**0.5 / tf.reduce_sum(update**2)**0.5,
            #     0.8)
            
            # tf.print(tf.reduce_sum(geoUpdate**2)**0.5 / tf.reduce_sum(update**2)**0.5)

            #Assign the update
            XUpdate = X + update[:,0] #+ tf.cast(geoCondition, DATATYPE)*geoUpdate[:,0]
                
            #Calculate a new loss
            lossUpdate = y - self.transformFunction(x,XUpdate)
            
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
                X = XUpdate

                #Calculate new jacobian
                J = self.jacobianFunction(x,X)
                              
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
        
        loss = y - self.transformFunction(x,X)

        return X, J, loss

#%%        
    def weightedTrain(self,x,y,X,W,dampingFactor,DISPLAY,FAILURE_COUNT_MAX,CHANGE_MIN,ITERATION_MAX):
        
        print("Tracing: Train")                
        optimised = tf.constant(False, dtype = tf.bool)
        
        failureCount = 0
                 
        epoch = tf.constant(0, dtype = tf.int32)
        
        #Initialise first loss and jacobian
        loss = W@(y - self.transformFunction(x,X))
        lossSum = tf.math.reduce_sum(loss**2)
        J = self.jacobianFunction(x,X)
        
        lossSumChange = tf.constant(1,dtype = DATATYPE)
                
        while not optimised:
            
            JtJ = self.weightedJtJFunction(J,W,dampingFactor)

            # update = tf.linalg.lstsq(JtJ, tf.transpose(J) @ loss, fast=False)
            update = tf.linalg.inv(JtJ) @ tf.transpose(J) @ loss
                                               
            #Assign the update
            XUpdate = X + update[:,0]
            
            #Calculate a new loss
            lossUpdate = W@(y - self.transformFunction(x,XUpdate))
            
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
                X = XUpdate

                #Calculate new jacobian
                J = self.jacobianFunction(x,X)
                              
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
        
        loss = y - self.transformFunction(x,X)

        return X, J, loss

#%%    
    @tf.function
    def JtJFunction(self,J,damping_factor):
    
        print("Tracing: JtJFunction")
                      
        JtJ = tf.transpose(J) @ J
                
        JtJ  = JtJ + damping_factor * tf.linalg.tensor_diag_part(JtJ)*tf.eye(J.shape[1], dtype = DATATYPE)
        
        return JtJ
    
    @tf.function
    def weightedJtJFunction(self,J,W,damping_factor):
    
        print("Tracing: weightedJtJFunction")
                      
        JtJ = tf.transpose(J) @ W @ J
                
        JtJ  = JtJ + damping_factor * tf.linalg.tensor_diag_part(JtJ)*tf.eye(J.shape[1], dtype = DATATYPE)
        
        return JtJ
            
    @tf.function
    def geodesicAcceleration(self,x,X,update,J):
        
        print("Tracing: geodesicAcceleration")
        
        h = 0.1
        
        function = self.transformFunction(x,X)
        functionUpdate = tf.reshape(self.transformFunction(x,X+tf.squeeze(h*update)), (-1,1))
        
        dirDiv = 2/h * ((functionUpdate - function)/h - J@update)
        
        return dirDiv
    


