import tensorflow as tf

class Calibration():
#%%   
    def __init__(self, datatype=tf.float64):
        self.datatype = datatype
        self.init_base_functions()

        self.options = {}
        self.options["damping_factor"] = 10
        self.options["verbosity"] = 1
        self.options["max_failure"] = 5
        self.options["min_change"] = 1e-10
        self.options["max_iterations"] = 500
        
    def init_base_functions(self):
        
        print("Tracing...", end="")
        
        self.rodrigues_TF = tf.function(self.rodrigues,
            input_signature=(tf.TensorSpec(shape=[3], dtype=self.datatype),))
        self.rodrigues_TF = self.rodrigues_TF.get_concrete_function()

        self.rodriguesinv_TF = tf.function(self.rodriguesinv,
            input_signature=(tf.TensorSpec(shape=[3,3], dtype=self.datatype),))
        self.rodriguesinv_TF = self.rodriguesinv_TF.get_concrete_function()

        self.assemble_camera_matrix_TF = tf.function(self.assemble_camera_matrix,
            input_signature=(tf.TensorSpec(shape=[5], dtype=self.datatype),))
        self.assemble_camera_matrix_TF = self.assemble_camera_matrix_TF.get_concrete_function()

        self.JtJ_TF = tf.function(self.JtJ)

        # self.weighted_JtJ_TF = tf.function(self.weighted_JtJ)
        # self.weighted_JtJ_TF = self.weighted_JtJ_TF.get_concrete_function()
        print("done")

    def rodrigues(self,r):
            
        theta = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(r)))
        
        zero = tf.zeros((1), dtype = self.datatype)
        
        r_norm = r/theta
        
        r1,r2,r3 = tf.split(r_norm, 3)
        
        K = tf.concat([
            tf.concat([zero, tf.negative(r3), r2], 0),
            tf.concat([r3, zero, tf.negative(r1)], 0),
            tf.concat([tf.negative(r2), r1, zero], 0),
            ],0)
        
        K = tf.reshape(K, (3,3))
    
        R = tf.eye(3, dtype = self.datatype) + tf.sin(theta)*K + (1-tf.cos(theta))*K@K
        
        return R
    
    def rodriguesinv(self,R):
        
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
        
    def assemble_camera_matrix(self,k):
                    
        fx, fy, s, u0, v0 = tf.split(k,5)
        
        zero = tf.zeros((1), dtype = self.datatype)
        one = tf.ones((1), dtype = self.datatype)
        
        K = tf.concat([
            tf.concat([fx, s, u0], 0),
            tf.concat([zero, fy, v0], 0),
            tf.concat([zero, zero, one], 0),
            ],0)
        
        K = tf.reshape(K, (3,3))
        
        return K
                                        
    def print_update(self, epoch, damping_factor, loss_sum):

        tf.print(
            "Epoch: ", epoch,
            ", lambda: ", damping_factor,
            ", loss: ", loss_sum)
    
    def iterate_damping(self, damping_factor, change):
        
        damping_factor = damping_factor*change
        
        if damping_factor<1e-5: damping_factor = damping_factor*0 + 1e-5
        
        return damping_factor

    def get_options(self):

        options = (
            self.options["damping_factor"],
            self.options["verbosity"],
            self.options["max_failure"],
            self.options["min_change"],
            self.options["max_iterations"]
        )
        return options

    def JtJ(self,J,damping_factor):
                          
        JtJ = tf.transpose(J) @ J
                
        JtJ  = JtJ + damping_factor * tf.linalg.tensor_diag_part(JtJ)*tf.eye(J.shape[1], dtype = self.datatype)
        
        return JtJ
    
    def weighted_JtJ(self,J,W,damping_factor):
                         
        JtJ = tf.transpose(J) @ W @ J
                
        JtJ  = JtJ + damping_factor * tf.linalg.tensor_diag_part(JtJ)*tf.eye(J.shape[1], dtype = DATATYPE)
        
        return JtJ
                      
    def train(self,x,y,X):
                    
        optimised = tf.constant(False, dtype = tf.bool)
        damping_factor,verbosity,max_failure,min_change, max_iterations = self.get_options()
        failureCount = 0
                 
        epoch = tf.constant(0, dtype = tf.int32)
        
        #Initialise first loss and jacobian
        loss = y - self.transform_TF(x,X)
        lossSum = tf.math.reduce_sum(loss**2)
        J = self.jacobian_TF(x,X)
        
        lossSumChange = tf.constant(1,dtype = self.datatype)
                
        while not optimised:
            
            JtJ = self.JtJ_TF(J,damping_factor)

            #Solve the linear system
            update = tf.linalg.inv(JtJ) @ tf.transpose(J) @ loss
            
            #Assign the update
            XUpdate = X + update[:,0] 

            #Calculate a new loss
            lossUpdate = y - self.transform_TF(x,XUpdate)
            
            #Has this improved the loss?
            lossSumUpdate = tf.reduce_sum(lossUpdate**2)
                                 
            lossSumChange = lossSumUpdate - lossSum
            
            condition = tf.math.less(lossSumChange, 0)

            #If condition is True
            if condition:
                 
                #Decrease the damping
                damping_factor = self.iterate_damping(damping_factor, 0.5)
                
                #Accept new value of loss, loss sum and parameters
                loss = lossUpdate
                lossSum = lossSumUpdate
                X = XUpdate

                #Calculate new jacobian
                J = self.jacobian_TF(x,X)
                              
                #Reset consecutive failure count
                failureCount = 0
                                                                
            #If condition2 fails    
            else:
                
                #Increase the damping
                damping_factor = self.iterate_damping(damping_factor, 5)
                failureCount = failureCount + 1
                                            
            #Optimisation Check
            optimised = (epoch>max_iterations) | (failureCount>max_failure) | ((lossSumChange<0)&(lossSumChange>-min_change))
            if verbosity == 1:self.print_update(epoch, damping_factor, lossSum)
            
            epoch = tf.add(epoch, 1)

        if verbosity != 0: 
            self.print_update(epoch, damping_factor, lossSum)            
            tf.print("\n===", "FINISHED", "===\n")
        
        loss = y - self.transform_TF(x,X)

        return X, J, loss
  
    def weightedTrain(self,x,y,X,W,damping_factor,DISPLAY,FAILURE_COUNT_MAX,CHANGE_MIN,ITERATION_MAX):
        
        print("Tracing: Train")                
        optimised = tf.constant(False, dtype = tf.bool)
        
        failureCount = 0
                 
        epoch = tf.constant(0, dtype = tf.int32)
        
        #Initialise first loss and jacobian
        loss = W@(y - self.transformFunction(x,X))
        lossSum = tf.math.reduce_sum(loss**2)
        J = self.jacobianFunction(x,X)
        
        lossSumChange = tf.constant(1,dtype = self.datatype)
                
        while not optimised:
            
            JtJ = self.weightedJtJFunction(J,W,damping_factor)

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
                damping_factor = self.iterateDamping(damping_factor, 0.5)
                
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
                damping_factor = self.iterateDamping(damping_factor, 5)
                failureCount = failureCount + 1
                                            
            #Optimisation Check
            optimised = (epoch>ITERATION_MAX) | (failureCount>FAILURE_COUNT_MAX) | ((lossSumChange<0)&(lossSumChange>-CHANGE_MIN))
            if DISPLAY == 1:self.printUpdate(epoch, damping_factor, lossSum)
            
            epoch = tf.add(epoch, 1)

        if DISPLAY != 0: 
            self.printUpdate(epoch, damping_factor, lossSum)            
            tf.print("\n===", "FINISHED", "===\n")
        
        loss = y - self.transformFunction(x,X)

        return X, J, loss
    


