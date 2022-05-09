#%% Imports
import h5py
import time

from commonlib.h5py_functions import load_h5py_arrays as load
from commonlib.console_outputs import ProgressBar

#%%
class Decode():
   
    def __init__(self, useCUDA = False):
        
        self.useCUDA = useCUDA
        
        if self.useCUDA:
            import cupy as xp
            self.xp = xp
        else:
            import numpy as xp
            self.xp = xp
        
        self.R = Retreive(useCUDA)
        
    def toGPU(self,inCPU):
        xp = self.xp
        
        if isinstance(inCPU, list):
            for i in range(len(inCPU)):
                inCPU[i] = xp.array(inCPU[i])    
        else:
            return xp.array(inCPU) 
    
    def toCPU(self,inGPU):
        xp = self.xp
                
        for i in range(len(inGPU)):
            inGPU[i] = xp.asnumpy(inGPU[i]) 
        
    def phase_plus_binary(self,measurementDirectory, regimeDirectory, freq, use_image_mask = True, phaseQualityMin = 0.3, offsetIndex=0):
        
        xp = self.xp
        
        bar = ProgressBar()
        
        barMax = 4
        bar.updateBar(0,barMax, suffix='Solving phase...')
            
        #LUT
        LUT = load(regimeDirectory,"LUT")
        if self.useCUDA:LUT = self.toGPU(LUT)
        
        # Solve for phase
        imageStack = load(measurementDirectory,(0+offsetIndex,10+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
            
        wrappedPhase, phaseQuality, averageIntensity, _ = self.R.nStepPhase(imageStack)
        
        #Change range to be 0-1 instead of -pi/2 - pi/2
        wrappedPhase = wrappedPhase/(xp.pi*2)
        i = wrappedPhase<0
        wrappedPhase[i] = wrappedPhase[i] + 1
                        
        bar.updateBar(1,barMax, suffix='Solving binary...')
        
        # Solve binary
        imageStack = load(measurementDirectory,(10+offsetIndex,16+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
    
        k_array = self.R.binary(imageStack, LUT, averageIntensity)
                    
        bar.updateBar(2,barMax, suffix='Unwrapping...')
        
        # Unwrap
        unwrappedPhase = wrappedPhase + k_array
        unwrappedPhase = unwrappedPhase / (freq)
        
        bar.updateBar(3,barMax, suffix='Solving mask...')
        
        # obtain through mask
                        
        mask = phaseQuality<phaseQualityMin
        
        if use_image_mask:
            
            maskImage = load(measurementDirectory,16+offsetIndex)
            if self.useCUDA:maskImage = self.toGPU(maskImage)
            
            mask2 = maskImage>averageIntensity
            
            mask = xp.logical_or(mask, mask2)
                                          
        bar.updateBar(4,barMax)
    
        #Return map    
        unwrappedPhase[mask] = 0
        
        return unwrappedPhase
    
    def phase_plus_binary_calibration(self,measurementDirectory,regimeDirectory, freq, use_image_mask = True, phaseQualityMin = 0.3):
                
        unwrappedPhaseX = self.phasePlusBinary(measurementDirectory,
                                               regimeDirectory, 
                                               freq,
                                               use_image_mask = True, 
                                               phaseQualityMin = phaseQualityMin, 
                                               offsetIndex=0)
        unwrappedPhaseY = self.phasePlusBinary(measurementDirectory, 
                                               regimeDirectory, 
                                               freq,
                                               use_image_mask = True, 
                                               phaseQualityMin = phaseQualityMin, 
                                               offsetIndex=16)

        return unwrappedPhaseX, unwrappedPhaseY
    
    def heterodyne5step(self,measurementDirectory, freq, offsetIndex=0):
        
        assert isinstance(freq, tuple) | isinstance(freq, list)
        assert len(freq)==3
        
        #Solve for heterodyne fringes
        p1 = 1/freq[0]
        p2 = 1/freq[1]
        p3 = 1/freq[2]
                
        p12 = p1 * p2 / (p2 - p1)
        p23 = p2 * p3 / (p3 - p2)
        
        p123 = p12 * p23 / (p23 - p12)

        bar = ProgressBar()
        
        barMax = 2
        bar.updateBar(0,barMax, suffix='Solving phase...')
                        
        # Solve for phase
        imageStack = load(measurementDirectory,(0+offsetIndex,5+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
        Dn1 = self.R.Hariharan(imageStack)
        
        imageStack = load(measurementDirectory,(5+offsetIndex,10+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
        Dn2 = self.R.Hariharan(imageStack)
        
        imageStack = load(measurementDirectory,(10+offsetIndex,15+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
        Dn3 = self.R.Hariharan(imageStack)
        
        # Obtain fractional orders
        bar.updateBar(1,barMax, suffix='Unwrapping...')
        
        unwrappedPhase = self.__heterodyneCombiner(Dn1,Dn2,Dn3,p1,p12,p123)
        
        bar.updateBar(2,barMax)
        
        return unwrappedPhase

    def heterodyne4step(self,measurementDirectory, freq, offsetIndex=0):
        
        assert isinstance(freq, tuple) | isinstance(freq, list)
        assert len(freq)==3
        
        #Solve for heterodyne fringes
        p1 = 1/freq[0]
        p2 = 1/freq[1]
        p3 = 1/freq[2]
                
        p12 = p1 * p2 / (p2 - p1)
        p23 = p2 * p3 / (p3 - p2)
        
        p123 = p12 * p23 / (p23 - p12)

        bar = ProgressBar()
        
        barMax = 2
        bar.updateBar(0,barMax, suffix='Solving phase...')
                        
        # Solve for phase
        imageStack = load(measurementDirectory,(0+offsetIndex,4+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
        Dn1 = self.R.Carre(imageStack)
        
        imageStack = load(measurementDirectory,(4+offsetIndex,8+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
        Dn2 = self.R.Carre(imageStack)
        
        imageStack = load(measurementDirectory,(8+offsetIndex,12+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
        Dn3 = self.R.Carre(imageStack)
        
        # Obtain fractional orders
        bar.updateBar(1,barMax, suffix='Unwrapping...')
        
        unwrappedPhase = self.__heterodyneCombiner(Dn1,Dn2,Dn3,p1,p12,p123)
        
        bar.updateBar(2,barMax)
        
        return unwrappedPhase
    
    def heterodyne3step(self,measurementDirectory, freq, offsetIndex=0):
        
        assert isinstance(freq, tuple) | isinstance(freq, list)
        assert len(freq)==3
        
        #Solve for heterodyne fringes
        p1 = 1/freq[0]
        p2 = 1/freq[1]
        p3 = 1/freq[2]
                
        p12 = p1 * p2 / (p2 - p1)
        p23 = p2 * p3 / (p3 - p2)
        
        p123 = p12 * p23 / (p23 - p12)

        bar = ProgressBar()
        
        barMax = 2
        bar.updateBar(0,barMax, suffix='Solving phase...')
                        
        # Solve for phase
        imageStack = load(measurementDirectory,(0+offsetIndex,3+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
        Dn1 = self.R.threeStep(imageStack)
        
        imageStack = load(measurementDirectory,(3+offsetIndex,6+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
        Dn2 = self.R.threeStep(imageStack)
        
        imageStack = load(measurementDirectory,(6+offsetIndex,9+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
        Dn3 = self.R.threeStep(imageStack)
        
        # Obtain fractional orders
        bar.updateBar(1,barMax, suffix='Unwrapping...')
        
        unwrappedPhase = self.__heterodyneCombiner(Dn1,Dn2,Dn3,p1,p12,p123)
        
        bar.updateBar(2,barMax)
        
        return unwrappedPhase

    def __heterodyneCombiner(self,Dn1,Dn2,Dn3,p1,p12,p123):
        
        xp = self.xp
        
        Dn1 = Dn1/(2*xp.pi)
        Dn2 = Dn2/(2*xp.pi)
        Dn3 = Dn3/(2*xp.pi)
                
        i = Dn1<Dn2
        j = Dn1>=Dn2
        Dn12 = xp.empty_like(Dn1)
        Dn12[i] = Dn1[i] - Dn2[i] + 1
        Dn12[j] = Dn1[j] - Dn2[j] 
        
        i = Dn2<Dn3
        j = Dn2>=Dn3
        Dn23 = xp.empty_like(Dn1)
        Dn23[i] = Dn2[i] - Dn3[i] + 1
        Dn23[j] = Dn2[j] - Dn3[j] 
        
        i = Dn12<Dn23
        j = Dn12>=Dn23
        Dn123 = xp.empty_like(Dn1)
        Dn123[i] = Dn12[i] - Dn23[i] + 1
        Dn123[j] = Dn12[j] - Dn23[j] 
        
        # obtain interger orders
        N12 = xp.around(p123*Dn123/p12 - Dn12)
        N1 = xp.around(p12*(N12 + Dn12)/p1 - Dn1)
        
        #Combine
        unwrappedPhase = (N1+Dn1)*p1
        
        return unwrappedPhase
        
    #%% modified heterodyne 
    
    def modifiedHeterodyne5stepFilter(self,measurementDirectory,freq,offsetIndex=0,r=0.3):
        
        assert isinstance(freq, tuple) | isinstance(freq, list)
        assert len(freq)==3
        try:
            assert num_of_keys(measurementDirectory) == 31, \
                "number of keys should be 31, but is {}".format(num_of_keys(measurementDirectory))            
        except Exception as e:
            print(e)


        #Solve for heterodyne fringes
        p1 = 1/freq[0]
        p2 = 1/freq[1]
        p3 = 1/freq[2]
                
        p12 = p1 * p2 / (p2 - p1)
        p23 = p2 * p3 / (p3 - p2)
        
        p123 = p12 * p23 / (p23 - p12)

        bar = ProgressBar()
        
        barMax = 2
        bar.updateBar(0,barMax, suffix='Solving phase...')
                        
        mask = self.createMask(r)
                
        # Solve for phase
        imageStack = self.loadFiltered(measurementDirectory,mask,(0+offsetIndex,5+offsetIndex))
        Dn1 = self.R.Hariharan(imageStack)
        
        imageStack = self.loadFiltered(measurementDirectory,mask,(5+offsetIndex,10+offsetIndex))
        Dn12 = self.R.Hariharan(imageStack)
        
        imageStack = self.loadFiltered(measurementDirectory,mask,(10+offsetIndex,15+offsetIndex))
        Dn123 = self.R.Hariharan(imageStack)
        
        bar.updateBar(1,barMax, suffix='Unwrapping...')
        
        unwrappedPhase = self.__modifiedHeterodyneCombiner(Dn1,Dn12,Dn123,p1,p12,p123)
        
        bar.updateBar(2,barMax)
    
        return unwrappedPhase
    
    def modifiedHeterodyne5step(self,measurementDirectory,freq,offsetIndex=0):
        
        assert isinstance(freq, tuple) | isinstance(freq, list)
        assert len(freq)==3
        try:
            assert num_of_keys(measurementDirectory) == 31, \
                "number of keys should be 31, but is {}".format(num_of_keys(measurementDirectory))            
        except Exception as e:
            print(e)


        
        #Solve for heterodyne fringes
        p1 = 1/freq[0]
        p2 = 1/freq[1]
        p3 = 1/freq[2]
                
        p12 = p1 * p2 / (p2 - p1)
        p23 = p2 * p3 / (p3 - p2)
        
        p123 = p12 * p23 / (p23 - p12)

        bar = ProgressBar()
        
        barMax = 2
        bar.updateBar(0,barMax, suffix='Solving phase...')
                        
        # Solve for phase
        imageStack = load(measurementDirectory,(0+offsetIndex,5+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
        Dn1 = self.R.Hariharan(imageStack)
        
        imageStack = load(measurementDirectory,(5+offsetIndex,10+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
        Dn12 = self.R.Hariharan(imageStack)
        
        imageStack = load(measurementDirectory,(10+offsetIndex,15+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
        Dn123 = self.R.Hariharan(imageStack)
        
        bar.updateBar(1,barMax, suffix='Unwrapping...')
        
        unwrappedPhase = self.__modifiedHeterodyneCombiner(Dn1,Dn12,Dn123,p1,p12,p123)
        
        bar.updateBar(2,barMax)
    
        return unwrappedPhase
    
    def modifiedHeterodyne4step(self,measurementDirectory,freq,offsetIndex=0):
        
        assert isinstance(freq, tuple) | isinstance(freq, list)
        assert len(freq)==3
        assert num_of_keys(measurementDirectory) == 31
        
        #Solve for heterodyne fringes
        p1 = 1/freq[0]
        p2 = 1/freq[1]
        p3 = 1/freq[2]
                
        p12 = p1 * p2 / (p2 - p1)
        p23 = p2 * p3 / (p3 - p2)
        
        p123 = p12 * p23 / (p23 - p12)

        bar = ProgressBar()
        
        barMax = 2
        bar.updateBar(0,barMax, suffix='Solving phase...')
                        
        # Solve for phase
        imageStack = load(measurementDirectory,(0+offsetIndex,4+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
        Dn1 = self.R.Carre(imageStack)
        
        imageStack = load(measurementDirectory,(4+offsetIndex,8+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
        Dn12 = self.R.Carre(imageStack)
        
        imageStack = load(measurementDirectory,(8+offsetIndex,12+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
        Dn123 = self.R.Carre(imageStack)
        
        bar.updateBar(1,barMax, suffix='Unwrapping...')
        
        unwrappedPhase = self.__modifiedHeterodyneCombiner(Dn1,Dn12,Dn123,p1,p12,p123)
        
        bar.updateBar(2,barMax)
    
        return unwrappedPhase
    
    def modifiedHeterodyne3stepFilter(self,measurementDirectory,freq,offsetIndex=0,r=0.3,subSample=1):
        
        assert isinstance(freq, tuple) | isinstance(freq, list)
        assert len(freq)==3
        assert num_of_keys(measurementDirectory) == 19
        
        #Solve for heterodyne fringes
        p1 = 1/freq[0]
        p2 = 1/freq[1]
        p3 = 1/freq[2]
                
        p12 = p1 * p2 / (p2 - p1)
        p23 = p2 * p3 / (p3 - p2)
        
        p123 = p12 * p23 / (p23 - p12)

        bar = ProgressBar()
        
        barMax = 2
        bar.updateBar(0,barMax, suffix='Solving phase...')
        
        mask = self.createMask(r)

        # Solve for phase
        imageStack = self.loadFiltered(measurementDirectory,mask,(0+offsetIndex,3+offsetIndex),subSample)
        Dn1 = self.R.threeStep(imageStack)
        
        imageStack = self.loadFiltered(measurementDirectory,mask,(3+offsetIndex,6+offsetIndex),subSample)
        Dn12 = self.R.threeStep(imageStack)
        
        imageStack = self.loadFiltered(measurementDirectory,mask,(6+offsetIndex,9+offsetIndex),subSample)
        Dn123 = self.R.threeStep(imageStack)
        
        bar.updateBar(1,barMax, suffix='Unwrapping...')
        
        unwrappedPhase = self.__modifiedHeterodyneCombiner(Dn1,Dn12,Dn123,p1,p12,p123)
        
        bar.updateBar(2,barMax)
    
        return unwrappedPhase
    
    def modifiedHeterodyne3step(self,measurementDirectory,freq,offsetIndex=0, subSample=1):
        
        assert isinstance(freq, tuple) | isinstance(freq, list)
        assert len(freq)==3
        assert num_of_keys(measurementDirectory) == 19
        
        #Solve for heterodyne fringes
        p1 = 1/freq[0]
        p2 = 1/freq[1]
        p3 = 1/freq[2]
                
        p12 = p1 * p2 / (p2 - p1)
        p23 = p2 * p3 / (p3 - p2)
        
        p123 = p12 * p23 / (p23 - p12)

        bar = ProgressBar()
        
        barMax = 2
        bar.updateBar(0,barMax, suffix='Solving phase...')
                        
        # Solve for phase
        imageStack = self.load(measurementDirectory,(0+offsetIndex,3+offsetIndex), subSample)
        Dn1 = self.R.threeStep(imageStack)
        
        imageStack = self.load(measurementDirectory,(3+offsetIndex,6+offsetIndex), subSample)
        Dn12 = self.R.threeStep(imageStack)
        
        imageStack = self.load(measurementDirectory,(6+offsetIndex,9+offsetIndex), subSample)
        Dn123 = self.R.threeStep(imageStack)
        
        bar.updateBar(1,barMax, suffix='Unwrapping...')
        
        unwrappedPhase = self.__modifiedHeterodyneCombiner(Dn1,Dn12,Dn123,p1,p12,p123)
        
        bar.updateBar(2,barMax)
    
        return unwrappedPhase
    
    def __modifiedHeterodyneCombiner(self,Dn1,Dn12,Dn123,p1,p12,p123):
        
        xp=self.xp
        
        Dn1 = Dn1/(2*xp.pi)
        Dn12 = Dn12/(2*xp.pi)
        Dn123 = Dn123/(2*xp.pi)
        
        #Change range to be 0-pi instead of -pi/2 - pi/2
        i = Dn123<0
        Dn123[i] = Dn123[i] + 1

        # obtain interger orders
        N12 = xp.around(p123*Dn123/p12 - Dn12)
        N1 = xp.around(p12*(N12 + Dn12)/p1 - Dn1)
        
        #Combine
        unwrappedPhase = (N1+Dn1)*p1

        return unwrappedPhase
            
    def testFilteredImages(self,measurementDirectory):
        xp = self.xp
        offsetIndex = 0
        
        imageStack = load(measurementDirectory,(0+offsetIndex,4+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
                
        import matplotlib.pyplot as plt
        plt.close('all')
        plt.figure()
        plt.imshow(imageStack[1].get())
        
        plt.figure()
        plt.plot(imageStack[1].get()[3000,:])
            
        X = imageStack[1]
        
        mask = self.createMask(0.3)
        
        FT = xp.fft.fftshift(xp.fft.fft2(xp.fft.fftshift(X)))
        
        X1 = xp.fft.fftshift(xp.fft.ifft2(xp.fft.fftshift(FT * mask)))
        
        plt.figure()
        plt.imshow(FT.real.get())
        
        plt.figure()
        plt.imshow(X1.real.get())
        
        plt.figure()
        plt.plot(X1.real.get()[3000,:])
        
        plt.figure()
        plt.imshow(mask.get())
        
    def loadFiltered(self,measurementDirectory,mask,index,subSample=1):
        xp = self.xp
        
        imageStack = self.load(measurementDirectory,index,subSample)
        if self.useCUDA:self.toGPU(imageStack)
        
        for i in range(len(imageStack)):
                
            X = imageStack[i]
            FT = xp.fft.fftshift(xp.fft.fft2(xp.fft.fftshift(X)))
            X1 = xp.fft.fftshift(xp.fft.ifft2(xp.fft.fftshift(FT * mask)))
            X1 = xp.abs(X1)
            
            imageStack[i] = X1
            
        return imageStack
    
    def load(self,array,indices,subSample=1):
        
        out = []
        
        start = indices[0]
        end = indices[1]
        with h5py.File(array, 'r') as f:
            for i in range(start,end):
                
                string = "{:02d}".format(i)
                               
                out.append(f[string][()][::subSample,::subSample])
        
        if self.useCUDA:self.toGPU(out)
        
        return out
               
    def createMask(self,r):
        xp = self.xp
        x = xp.linspace(-1,1,5120).reshape(-1,1)
        y = xp.linspace(-1,1,5120).reshape(1,-1)
        
        mask = xp.exp(-(x**2 + y**2)/r**2)
        return mask
        
    def showMask(self,r):
        xp = self.xp
        x = xp.linspace(-1,1,5120)
        
        xFT = xp.fft.fftshift(xp.fft.fftfreq(5120, 1/5120))
        
        y = xp.exp(-x**2/r**2)
                
        yFT = xp.fft.fftshift(xp.fft.fft(xp.fft.fftshift(y)))
        yFT = xp.abs(yFT)
        yFT = yFT/yFT.max()

        return (xFT,y), (xp.linspace(-2560,2559,5120),yFT)
    
    def filterImage(self,X, mask=None, r=0.4):
        
        xp = self.xp
        
        if mask is None: mask = self.createMask(r)
        
        FT = xp.fft.fftshift(xp.fft.fft2(xp.fft.fftshift(X)))
        X1 = xp.fft.fftshift(xp.fft.ifft2(xp.fft.fftshift(FT * mask)))
        X1 = xp.abs(X1)
        
        return X1
    
    def filterImages(self,X,mask):
        
        for i in range(len(X)):
            
            X[i] = self.filterImage(X[i])
        
        return X
    
    def modifiedHeterodyne4step(self,measurementDirectory,freq,offsetIndex=0):
        
        assert isinstance(freq, tuple) | isinstance(freq, list)
        assert len(freq)==3
        
        #Solve for heterodyne fringes
        p1 = 1/freq[0]
        p2 = 1/freq[1]
        p3 = 1/freq[2]
                
        p12 = p1 * p2 / (p2 - p1)
        p23 = p2 * p3 / (p3 - p2)
        
        p123 = p12 * p23 / (p23 - p12)

        bar = ProgressBar()
        
        barMax = 2
        bar.updateBar(0,barMax, suffix='Solving phase...')
                        
        # Solve for phase
        imageStack = load(measurementDirectory,(0+offsetIndex,4+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
        Dn1 = self.R.Carre(imageStack)
        
        imageStack = load(measurementDirectory,(4+offsetIndex,8+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
        Dn12 = self.R.Carre(imageStack)
        
        imageStack = load(measurementDirectory,(8+offsetIndex,12+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
        Dn123 = self.R.Carre(imageStack)
        
        bar.updateBar(1,barMax, suffix='Unwrapping...')
        
        unwrappedPhase = self.__modifiedHeterodyneCombiner(Dn1,Dn12,Dn123,p1,p12,p123)
        
        bar.updateBar(2,barMax)
    
        return unwrappedPhase
    
    def modifiedHeterodyne3step(self,measurementDirectory,freq,offsetIndex=0):
        
        assert isinstance(freq, tuple) | isinstance(freq, list)
        assert len(freq)==3
        
        #Solve for heterodyne fringes
        p1 = 1/freq[0]
        p2 = 1/freq[1]
        p3 = 1/freq[2]
                
        p12 = p1 * p2 / (p2 - p1)
        p23 = p2 * p3 / (p3 - p2)
        
        p123 = p12 * p23 / (p23 - p12)

        bar = ProgressBar()
        
        barMax = 2
        bar.updateBar(0,barMax, suffix='Solving phase...')
                        
        # Solve for phase
        imageStack = load(measurementDirectory,(0+offsetIndex,3+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
        Dn1 = self.R.threeStep(imageStack)
        
        imageStack = load(measurementDirectory,(3+offsetIndex,6+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
        Dn12 = self.R.threeStep(imageStack)
        
        imageStack = load(measurementDirectory,(6+offsetIndex,9+offsetIndex))
        if self.useCUDA:self.toGPU(imageStack)
        Dn123 = self.R.threeStep(imageStack)
        
        bar.updateBar(1,barMax, suffix='Unwrapping...')
        
        unwrappedPhase = self.__modifiedHeterodyneCombiner(Dn1,Dn12,Dn123,p1,p12,p123)
        
        bar.updateBar(2,barMax)
    
        return unwrappedPhase
    
    def __modifiedHeterodyneCombiner(self,Dn1,Dn12,Dn123,p1,p12,p123):
        
        xp=self.xp
        
        Dn1 = Dn1/(2*xp.pi)
        Dn12 = Dn12/(2*xp.pi)
        Dn123 = Dn123/(2*xp.pi)
        
        #Change range to be 0-pi instead of -pi/2 - pi/2
        i = Dn123<0
        Dn123[i] = Dn123[i] + 1

        # obtain interger orders
        N12 = xp.around(p123*Dn123/p12 - Dn12)
        N1 = xp.around(p12*(N12 + Dn12)/p1 - Dn1)
        
        #Combine
        unwrappedPhase = (N1+Dn1)*p1

        return unwrappedPhase
        
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

    def Hariharan(self, I):
        xp = self.xp
        
        top = 2*(I[1].astype(float) - I[3].astype(float))
        bottom =  2*I[2].astype(float) - I[0].astype(float) - I[4].astype(float)
        out = xp.arctan2(top,bottom, dtype=float) 
        
        return out
    
    def HariharanQuality(self,I):
        
        A = I[0] + I[1] + 2*I[2] + I[3] + I[4]
        B = (4*(I[1] - I[3])**2 + (2*I[2] - I[0] - I[4])**2)**0.5
        
        return (6*B)/(4*A)

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
