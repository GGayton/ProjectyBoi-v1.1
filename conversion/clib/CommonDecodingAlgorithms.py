class Retreive():
    
    def __init__(self, useCUDA = False):
        self.useCUDA = useCUDA
        
        if self.useCUDA:
            import cupy as xp
            self.xp = xp
        else:
            import numpy as xp
            self.xp = xp

    def nStepPhase(self,imgStack):
        
        xp = self.xp
    
        #Number of images
        N = len(imgStack)
            
        #Phase shift
        delta = xp.linspace(1,N,N)*2*xp.pi/N
    
        partA = xp.zeros(imgStack[0].shape, dtype = xp.float64)
        partB = xp.zeros(imgStack[0].shape, dtype = xp.float64)
        averageIntensity = xp.zeros(imgStack[0].shape, dtype = xp.float64)
        for i in range(N):
            partA = partA + imgStack[i]*xp.sin(delta[i])
            partB = partB + imgStack[i]*xp.cos(delta[i])
            averageIntensity = averageIntensity + imgStack[i]
    
        wrappedPhase = -xp.arctan2(partA, partB)
        
        averageIntensity = averageIntensity/N
        
        modulation = xp.sqrt(partA**2 + partB**2)/N
        
        phaseQuality = modulation/averageIntensity
        
        return wrappedPhase, phaseQuality, averageIntensity, modulation
    
    def binary(self,img_stack, LUT, averageIntensity):
        
        xp = self.xp
        
        #Create the binary array
        binary_array = xp.dstack(img_stack)
                
        binary_array = xp.greater(binary_array,averageIntensity.reshape(averageIntensity.shape[0], averageIntensity.shape[1],1))
               
        #Preallocate the k_array
        k_array = xp.full(img_stack[0].shape, 0, dtype=xp.int)
        
        N_codewords = LUT.shape[1]-1
        
        for i in range(LUT.shape[0]):
                        
            binary_check = LUT[i,1:].reshape(1,1,N_codewords)
                                   
            k_array[xp.all(binary_array==binary_check, axis=2)] = LUT[i,0]
            
        return k_array

    def Hariharan(self, imgStack):
        xp = self.xp
        
        I = imgStack
        
        top = 2*(I[1].astype(float) - I[3].astype(float))
        bottom =  2*I[2].astype(float) - I[0].astype(float) - I[4].astype(float)
        out = xp.arctan2(top,bottom, dtype=float) 
        
        return out

    def Carre(self, imgStack):
        xp = self.xp
        
        I = imgStack
        
        top = I[1].astype(float) - I[3].astype(float)
        bottom =  I[0].astype(float) - I[2].astype(float)
        out = xp.arctan2(top,bottom, dtype=float) 

        return out
    
    def threeStep(self, imgStack):
        xp = self.xp
        
        I = imgStack
        
        top = xp.sqrt(3)*(I[0].astype(float) - I[2].astype(float))
        bottom =  2*I[1].astype(float) - I[0].astype(float) - I[2].astype(float)
        out = xp.arctan2(top,bottom, dtype=float) 

        return out
