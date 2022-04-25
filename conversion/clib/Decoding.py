import h5py

import time
from clib.CommonDecodingAlgorithms import (
    Retreive)

from clib.h5pyFunctions import load_h5py_arrays as load
from clib.consoleOutputs import ProgressBar

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
        
    def phasePlusBinary(self,measurementDirectory, regimeDirectory, freq, use_image_mask = True, phaseQualityMin = 0.3, offsetIndex=0):
        
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
    
    def phasePlusBinaryCalibration(self,measurementDirectory,regimeDirectory, freq, use_image_mask = True, phaseQualityMin = 0.3):
                
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
    
    #%% heterodyne
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
    def modifiedHeterodyne5step(self,measurementDirectory,freq,offsetIndex=0):
        
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
        
    #%%  
        
        
        
        
        
        