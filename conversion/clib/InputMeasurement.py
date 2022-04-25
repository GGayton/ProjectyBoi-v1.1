from clib.Common import Common
import numpy as np

class InputMeasurement(Common):  
    
    def __init__(self,cov=None,camRes=(5120,5120),useCUDA=False):
        super().__init__(useCUDA)
        
        self.useCUDA = useCUDA
        if useCUDA: import cupy as xp
        else: import numpy as xp
        
        self.C = xp.zeros((4,4))
        
        if cov is not None:
            self.setCovariance(cov)
            
        self.xp=xp
        
        self.corrMap = xp.zeros((camRes[0], camRes[1], 2))
        
        self.mapIndex = xp.ones(camRes,dtype = bool)
                
    def updateMapIndex(self,newIndex):
        xp = self.xp
        self.mapIndex = xp.logical_and(self.mapIndex, self.go(newIndex))
        self.formVectors()
                   
    def setXMeasurement(self, corrMapIn):
        self.corrMap[:,:,0] = corrMapIn
        self.updateMapIndex(corrMapIn!=0)
        self.formVectors()
        
    def setYMeasurement(self, corrMapIn):
        self.corrMap[:,:,1] = corrMapIn
        self.updateMapIndex(corrMapIn!=0)
        self.formVectors()
    
    def setCovariance(self, covIn):
        xp = self.xp
        self.cov = covIn
        self.C = xp.linalg.cholesky(self.cov)
       
    def setSubsetMaps(self,listIn):
        
        xp = self.xp
        
        self.subsetMapList = []
        
        for i in range(len(listIn)):
            
            self.subsetMapList.append(xp.array(listIn[i]))
        
    def updateMapIndexWithSubsetMaps(self):
        
        xp = self.xp
                
        newMapIndex = xp.zeros(self.mapIndex.shape, dtype = bool)
        
        for subsetMap in self.subsetMapList:
            
            newMapIndex = xp.logical_or(newMapIndex, xp.logical_and(self.mapIndex, subsetMap))
        
        self.mapIndex = newMapIndex
        self.formVectorSubset()
       
    def updateSubsetMapsWithVector(self, listIn):
        
        xp = self.xp
        
        for i in range(len(listIn)):
            
            self.subsetMapList[i][self.subsetMapList[i]] = xp.logical_and(
                self.subsetMapList[i][self.subsetMapList[i]],
                listIn[i])
              
    def formVectors(self):
        xp = self.xp
                        
        u,v = xp.meshgrid(
            xp.linspace(1,self.corrMap.shape[0],self.corrMap.shape[0]),
            xp.linspace(1,self.corrMap.shape[1],self.corrMap.shape[1]),
            indexing='ij')
               
        u = u[self.mapIndex].reshape(1,-1)
        v = v[self.mapIndex].reshape(1,-1)

        self.pixelVector = xp.concatenate((u,v,xp.ones_like(u)),axis=0)
        
        self.measurementVector = xp.empty_like(self.pixelVector)
        self.measurementVector[0:2,:] = self.corrMap[self.mapIndex,:].T
        self.measurementVector[2,:] = 1
        
        self.pixelVector = [self.pixelVector]
        self.measurementVector = [self.measurementVector]
        
    def formVectorSubset(self):
        xp = self.xp
                    
        u,v = xp.meshgrid(
            xp.linspace(1,self.corrMap.shape[0],self.corrMap.shape[0]),
            xp.linspace(1,self.corrMap.shape[1],self.corrMap.shape[1]),
            indexing='ij')
        
        self.pixelVector = []
        self.measurementVector = []
        
        for subsetMap in self.subsetMapList:
            uSub = u[subsetMap].reshape(-1)
            vSub = v[subsetMap].reshape(-1)
    
            pixelVectorTemp = xp.empty((3,uSub.shape[0]))
            pixelVectorTemp[0,:] = uSub
            pixelVectorTemp[1,:] = vSub
            pixelVectorTemp[2,:] = 1
            self.pixelVector.append(pixelVectorTemp)
            
            measurementVectorTemp = xp.empty_like(pixelVectorTemp)
            measurementVectorTemp[0:2,:] = self.corrMap[subsetMap,:].T
            measurementVectorTemp[2,:] = 1
            self.measurementVector.append(measurementVectorTemp)
                              
    def get(self):
        return self.pixelVector, self.measurementVector
        
    def getVirtual(self):
        
        xp = self.xp
        
        pixelVector = self.pixelVector.copy()
        measurementVector = self.measurementVector.copy()
        
        for i in range(len(pixelVector)):
        
            randomPerturb = self.C @ xp.random.randn(4,pixelVector[i].shape[1])
            
            pixelVector[i][:2,:] = pixelVector[i][:2,:] + randomPerturb[:2,:]
            
            measurementVector[i][:2,:] = measurementVector[i][:2,:] + randomPerturb[2:,:]

        return pixelVector, measurementVector
    
    def orderSubsetMapsWithCoords(self,coordsIn):
        xp = self.xp
               
        pixelVectors,_ = self.get()
        
        assert len(coordsIn)==len(pixelVectors)

        N = len(coordsIn)
        
        currentCentres = xp.empty((2,N))
        coords = xp.empty((N,2))
        for i in range(N):
            currentCentres[:,i] = xp.mean(pixelVectors[i],axis=1)[:2]
            coords[i,0] = coordsIn[i][0]
            coords[i,1] = coordsIn[i][1]
        
        distList = (coords[:,0:1]-currentCentres[0:1,:])**2 + (coords[:,1:2]-currentCentres[1:2,:])**2
        newSubsetList = [0 for i in range(25)]
        oldSubsetList = self.subsetMapList
        for k in range(N):
            
            i,j = xp.unravel_index(np.argmin(distList), (25,25))
            i = int(i)
            j = int(j)
            
            newSubsetList[i] = oldSubsetList[j]
            
            distList[:,j] = xp.inf
        
        self.subsetMapList = newSubsetList

        