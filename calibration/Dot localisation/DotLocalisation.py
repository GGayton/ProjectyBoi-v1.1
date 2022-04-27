import numpy as np
import cv2
import matplotlib.pyplot as plt

from commonlib.common_functions import (
    sampleImageGradient,
    extractRegion,
    interp2D, 
    rescale_to_uint8
    )

from commonlib.NonLinearSolver import NonLinearSolver
from commonlib.LinearSolver import LinearSolver
from defineEllipseJacobian import defineEllipseJacobian
from commonlib.console_outputs import ProgressBar
from scipy.stats import chi2
from commonlib.common_functions import (
    corrMatrixFromCovMatrix,
    plotErrorEllipse)

import time
# import sparse

class DotLocalisation():
    
    def __init__(self):
        
        self.options = {
            "numLines" : 50,
            "resLines" : 45,
            "lenLines" : 45
            }
        
        self.NLSoptions = {
            "iterationMax": 100,
            "failureCountMax": 3,
            "minimumChange": 1,
            "verbosity": 0
            }    
        
        self.LSoptions = {
            "verbosity":0
            }
        
        self.initialiseNLS()
        
        self.initialiseLS()
        
        self.JacobianFunction = defineEllipseJacobian()
        
    #%% Main
        
    def localise(self, blankImage):
        
        bar = ProgressBar()
        
        #Use cv2 to find board estimate
        points = self.extractCameraPoints(blankImage)
        
        numPoints = 184
                
        cParams = np.empty((numPoints,5))
        cV = np.empty((numPoints,5,5))
        sigma = np.empty(numPoints)
        sigmaStd = np.empty(numPoints)
        
        bar.updateBar(0,numPoints)
        
        for i in range(numPoints):
            
            dot = points[i,:]

            array,xAdd,yAdd = extractRegion(blankImage, dot, 50)
            
            x0,y0,a,b,T,Vout,sigma[i],sigmaStd[i] = self.localisePoint(array)
            
            cParams[i,:5] = x0+xAdd,y0+yAdd,a,b,T
            cV[i,:,:] = Vout
            
            bar.updateBar(i+1,numPoints)

        return cParams,cV,sigma,sigmaStd
    
    def infer(self,P,Vx,mappingX,mappingY):
        
        bar = ProgressBar()
        
        numPoints = 184
        
        pParams = np.empty((numPoints,2))
        pV = np.empty((numPoints,2,2))
        
        bar.updateBar(0, numPoints)

        for i in range(numPoints):
            
            x0,y0,a,b,T = P[i,:]

            I = self.outsideIndex(x0,y0,a+8,b+8,T).T

            J = np.concatenate((mappingX[I[:,0], I[:,1]].reshape(-1,1), mappingY[I[:,0], I[:,1]].reshape(-1,1)), axis=1)
            
            A,VA = self.estimateTransform(I,J)
            
            point = np.ones((1,3))
            point[0,:2] = P[i,:2]
            
            pParams[i,:] = point@A
            
            V = np.zeros((8,8))
            V[:2,:2] = Vx[i,:2,:2]
            V[2:,2:] = VA
            
            J = np.zeros((8,2))
            J[:2,:] = A[:2,:]
            J[2:5,0] = point[0,:]
            J[5:8,1] = point[0,:]
            
            pV[i,:,:] = J.T@V@J
            
            bar.updateBar(i+1, numPoints)
            
        return pParams, pV
               
    def extractCameraPoints(self,image, boardSize=(8,23), pointsNum=184):
                   
            #Subsample images
            scaleFactor = 4
            reducedImage = image[::scaleFactor, ::scaleFactor]
            reducedImage = rescale_to_uint8(reducedImage, max_value=reducedImage.max())
            
            #Find circles image
            ret, points = cv2.findCirclesGrid(reducedImage, boardSize, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
            
            assert ret
            
            #Flip the points because they flip the damn axis of the image
            points = np.flip(points[:,0,:], axis=1)
                
            #Blow back up to original size
            points = points * scaleFactor
            
            return points
    
    def localisePoint(self,array):
        
        gradient = sampleImageGradient(array)
            
        z,ix,iy = self.LSFsamples(gradient)
        
        x,y,Vxy,sigma,sigmaStd = self.refineLineSpreadFunctions(z,ix,iy)
                                                           
        index = self.cleanEllipse(x,y).reshape(-1)

        xi = x[index]
        yi = y[index]
        
        #Initial estimate to estimate uncertainty
        x0,y0,a,b,T = self.estimateLeastSquaresEllipse(xi, yi)
                                        
        #Iterate through
        x0n,y0n,an,bn,Tn,Vout = self.estimateTotalLeastSquaresEllipse(x,y,Vxy,index)
        
        return x0n,y0n,a,b,T,Vout,sigma,sigmaStd
        
    def Hessian(self, blankImage):
                       
        bar = ProgressBar()
        
        #Use cv2 to find board estimate
        points = self.extractCameraPoints(blankImage)
        
        numPoints = 184
        
        bar.updateBar(0,numPoints)
        
        J = [None for _ in range(184)]
        H = [None for _ in range(184)]
        loss = [None for _ in range(184)]
        for i in range(numPoints):
            
            dot = points[i,:]

            array,xAdd,yAdd = extractRegion(blankImage, dot, 50)
            
            J[i],H[i],loss[i] = self.localiseHessian(array)
            
            # J[i] = sparse.COO(J[i])
            # H[i] = sparse.COO(H[i])
            
            bar.updateBar(i+1,numPoints)

        return J,H,loss
    
    def localiseHessian(self,array):
        
        gradient = sampleImageGradient(array)
            
        z,ix,iy = self.LSFsamples(gradient)
        
        numLines = self.options["numLines"]
        resLines = self.options["resLines"]

        x = np.tile(np.linspace(0,1,resLines).reshape(-1,1), [numLines,1])
        
        y1 = z/z.max(axis=0)
        y = y1.T.reshape(-1,1)
        
        zSum = np.sum(z, axis=1)
        zSum[:20] = 0
        
        muEst = np.argmax(zSum)/z.shape[0]
        
        X = np.empty((1+numLines*2,1))
        X[0,0] = 0.1
        X[1::2,0] = muEst
        X[2::2,0] = 1
        
        params, J, loss =  self.NLS.solve(x,y,X)
        
        # H = np.zeros((params.shape[0], params.shape[0],x.shape[0]))
        
        # for i in range(numLines):
            
        #     xin = y1[:,i]
        #     Xin = X[[0,i*2+1, i*2+2]]
        #     h = self.hessianFunction(xin,Xin)
            
        #     ia = i*resLines
        #     ib = ia + resLines
            
        #     H[0,0,ia:ib] = h[0,0,:]
        #     H[i*2+1:i*2+3,0,ia:ib] = h[0,1:,:]
        #     H[0,i*2+1:i*2+3,ia:ib] = h[1:,0,:]
        #     H[i*2+1:i*2+3,i*2+1:i*2+3,ia:ib] = h[1:,1:,:]
            
        H =[]
        Jnew = []
        for i in range(numLines):
            
            xin = y1[:,i]
            Xin = X[[0,i*2+1, i*2+2]]
            H += [self.hessianFunction(xin,Xin)]
            
            ia = i*resLines
            ib = ia + resLines
            Jnew += [J[ia:ib, [0,2*i+1,2*i+2]]]
        
        H = np.dstack(H)
        Jnew = np.vstack(Jnew)
                        
        return Jnew,H,loss
        
    def hessianFunction(self,x, X):
        """
        sigma,mu,A
        """
        s = X[0]
        mu = X[1]
        A = X[2]
        
        f = A*np.exp( - (x-mu)**2/(2*s**2))
        
        H = np.zeros((3, 3,x.shape[0]))
        
        H[0,0] = -3*A*(-mu + x)**2*f/s**4 + A*(-mu + x)**4*f/s**6
        H[1,1] = -A*f/s**2 + A*(2*mu - 2*x)**2*f/(4*s**4)
        
        H[1,0] = H[0,1] = A*(2*mu - 2*x)*f/s**3 - A*(-mu + x)**2*(2*mu - 2*x)*f/(2*s**5)
        H[2,0] = H[0,2] = (-mu + x)**2*f/s**3
        H[1,2] = H[2,1] = -(2*mu - 2*x)*f/(2*s**2)
        
        
        return H
        
    #%% Refine border estimation
    
    def LSFsamples(self,gradient):
        
        numLines = self.options["numLines"]
        resLines = self.options["resLines"]
        lenLines = self.options["lenLines"]
               
        T = np.linspace(0,2*np.pi, numLines+1)
        T = T[:-1]
    
        centre = (np.array(gradient.shape) - 1)/2
        
        I = np.linspace(0,lenLines,resLines)
        
        z = np.empty((resLines,  numLines))
        ix = np.empty((resLines, numLines))
        iy = np.empty((resLines, numLines))
        
        for i in range(numLines):
            
            ix[:,i] =  I*np.cos(T[i]) + centre[0]
            iy[:,i] =  I*np.sin(T[i]) + centre[1]
    
            z[:,i] = interp2D(gradient,ix[:,i],iy[:,i])
    
        return z,ix,iy
    
    def refineLineSpreadFunctions(self,z,ix,iy):
        
        numLines = self.options["numLines"]
        resLines = self.options["resLines"]

        x = np.tile(np.linspace(0,1,resLines).reshape(-1,1), [numLines,1])
        
        y = z/z.max(axis=0)
        y = y.T.reshape(-1,1)
        
        zSum = np.sum(z, axis=1)
        zSum[:20] = 0
        
        muEst = np.argmax(zSum)/z.shape[0]
        
        X = np.empty((1+numLines*2,1))
        X[0,0] = 0.1
        X[1::2,0] = muEst
        X[2::2,0] = 1
        
        params, J, loss =  self.NLS.solve(x,y,X)

        #only consider data considered part of the gaussian
        k=2
        lowerBound = np.repeat(params[1::2,0] - k*params[0,0], [resLines])
        upperBound = np.repeat(params[1::2,0] + k*params[0,0], [resLines])
        index = np.logical_and(x[:,0]>lowerBound, x[:,0]<upperBound)
        
        Vparams = np.sum(loss[index]**2)/(np.sum(index) - (1+numLines*2)) * np.linalg.inv(J[index].T@J[index])
        
        resLimit = 2.35*params[0,0]
        
        Vmu = Vparams[1::2,:][:,1::2]
          
        Vmu += np.eye(Vmu.shape[0])*resLimit**2/12
        
        index = params[1::2,0]
        
        xnew = (ix[-1,:] - ix[0,:]) * index + ix[0,:]
        ynew = (iy[-1,:] - iy[0,:]) * index + iy[0,:]
               
        xnew = xnew.reshape(-1,1)
        ynew = ynew.reshape(-1,1)
        
        Vxy = self.estimateXYCov(ix,iy,Vmu)
        
        #Testing
        # z1 = z.T.flatten().copy()
        # z1[np.invert(index)] = 0
        # z1 = z1.reshape((z.shape[1], z.shape[0])).T
        
        # for i in range(z1.shape[1]):
        #     plt.plot(z1[:,i])
        
        return xnew, ynew, Vxy, params[0,0], Vparams[0,0]**0.5

    def estimateXYCov(self,ix,iy,Vpos):
                       
        N = Vpos.shape[0]
                
        i = np.arange(0,N*2,2)
        j = np.arange(0,N,1)
        
        J = np.zeros((N*2, N))
        
        J[i,j] = ix[-1,:] - ix[0,:]
        J[i+1,j] = iy[-1,:] - iy[0,:]

        V = J@Vpos@J.T    
        
        # index = np.diag(V)==0
        # index = np.nonzero(index)[0]
        
        # V[index, index] = 1e-10
        
        return V
        
    #%% RANSAC cleaning
    
    @staticmethod
    def rotate2D(xp,yp,T):
        return xp*np.cos(T) - yp*np.sin(T), xp*np.sin(T) + yp*np.cos(T)
    
    def closestPointOnEllipse(self,X,Y,x0,y0,a,b,T):
        
        #Rescale and typecast
        L = 20
        X = (X-x0).astype(np.complex128)/L
        Y = (Y-y0).astype(np.complex128)/L
        a = a.astype(np.complex128)/L
        b = b.astype(np.complex128)/L

        #Revert points back to T=0
        X,Y = self.rotate2D(X,Y,-T)
        
        #Solve        
        c = pow(a, 6) - 2 * pow(a, 4)*(b*b) + (a*a)*pow(b, 4) - pow(a, 4)*(X*X) -(a*a)*(b*b)*(Y*Y)
        d = a*a - b*b
        e = pow(a, 4) - 2 * (a*a)*(b*b) + pow(b, 4) - (a*a)*(X*X) - (b*b)*(Y*Y)
        f = -108*pow(a, 8)*pow(d, 4)*(X*X) + 108*pow(a,10)*(d*d)*pow(X, 4) + 108*pow(a, 6)*(d*d)*(X*X)*c + 2*pow(c,3)
        g = (pow(2, 1/3)*(e*e)) / (3*(a*a)*(X*X)*pow(f + pow(f*f - 4*pow(c,6), 1/2), 1/3))
        h = (pow(f + pow(f*f - 4*pow(c, 6), 1/2), 1 / 3)) / (3 * pow(2, 1/3)*pow(a, 6)*(X*X))
        i = e / (pow(a,4)*(X*X))
        j = c/(3*pow(a,6)*(X*X))
        k = d / ((a*a)*X)
        m = j + g + h
        
        if np.any(np.isnan(g)): return np.nan, np.nan
        
        X1 = 0.5*(k - np.sqrt(k*k - i + m) - np.sqrt(2*(k*k) - i - m - ((8*k*(k*k - (2 / (a*a)) - i) / (4*np.sqrt(k*k - i + m))))))
        X2 = 0.5*(k - np.sqrt(k*k - i + m) + np.sqrt(2*(k*k) - i - m - ((8*k*(k*k - (2 / (a*a)) - i) / (4*np.sqrt(k*k - i + m))))))
        X3 = 0.5*(k + np.sqrt(k*k - i + m) - np.sqrt(2*(k*k) - i - m + ((8*k*(k*k - (2 / (a*a)) - i) / (4*np.sqrt(k*k - i + m))))))
        X4 = 0.5*(k + np.sqrt(k*k - i + m) + np.sqrt(2*(k*k) - i - m + ((8*k*(k*k - (2 / (a*a)) - i) / (4*np.sqrt(k*k - i + m))))))
        
        Y1 = ((a*a)*X*X1 + (b*b) - (a*a)) / ((b*b)*Y)
        Y2 = ((a*a)*X*X2 + (b*b) - (a*a)) / ((b*b)*Y)
        Y3 = ((a*a)*X*X3 + (b*b) - (a*a)) / ((b*b)*Y)
        Y4 = ((a*a)*X*X4 + (b*b) - (a*a)) / ((b*b)*Y)
        
        xt = np.concatenate((1/X1,1/X2,1/X3,1/X4), axis=1)
        yt = np.concatenate((1/Y1,1/Y2,1/Y3,1/Y4), axis=1)
        
        #Find distance
        dist = np.sqrt(pow(xt - X, 2) + pow(yt - Y, 2))
        
        #Find index of plaves with 4? real solutions (numerical issue?)
        i0 = np.invert(np.all(np.abs(dist.imag)<1e-15, axis=1))

        #Filter out the imaginary solutions - keep in mind the numerical deficiencies
        index = np.argmax(np.abs(dist[i0].imag), axis=1)
        dist[i0,index] = np.inf + 0j
        index = np.argmax(np.abs(dist[i0].imag), axis=1)
        dist[i0,index] = np.inf + 0j

        #Select smallest distance values from remaining real solutions
        index = np.argmin(dist.real, axis=1)
        dist = dist.real
                
        xe = xt[np.arange(0,X.shape[0],1),index].real
        ye = yt[np.arange(0,X.shape[0],1),index].real
        
        #Rotate back to original
        X,Y = self.rotate2D(X,Y,T)
        xe,ye = self.rotate2D(xe,ye,T)
        
        #Rescale back up
        xe = xe * L + x0
        ye = ye * L + y0
        
        X = X.real*L + x0
        Y = Y.real*L + y0

        return xe,ye
    
    @staticmethod
    def pseudoRandom(I):
        
        a = np.random.randint(1,4)
        b = 5-a
        
        H = I.shape[0]//2
        
        return np.concatenate((np.random.choice(I[:H],a,False), np.random.choice(I[H:],b,False)))
    
    def cleanEllipse(self,x,y,iterations=100):
                
        I = np.arange(0,x.shape[0],1)
        
        bestDistSum = 0
        bestDist = 0
        bestXY = np.empty((2*x.shape[0],1))
        
        #RANSAC algorithm
        for i in range(iterations):
            
            index = self.pseudoRandom(I)
            
            xTemp = x[index]
            yTemp = y[index]
                    
            params = self.estimateLeastSquaresEllipse(xTemp,yTemp)
            
            if not np.isnan(params[0]):    
                xe,ye = self.closestPointOnEllipse(x,y,params[0],params[1],params[2],params[3],params[4])
                
                dist = ((x.flatten()-xe)**2 + (y.flatten()-ye)**2)**0.5
                
                passes = dist<0.2
                
                distSum = np.sum(passes)
                
                if distSum>bestDistSum:
                    bestDist = dist
                    bestDistSum = distSum
                    keepIndex = passes
                    
                    bestXY[0::2,0] = x.flatten()-xe
                    bestXY[1::2,0] = y.flatten()-ye

        #Remove ridiculous points
        criticalValue = bestDist[keepIndex].mean() + 3*bestDist[keepIndex].std()
        indexOut = bestDist<criticalValue
                                
        return indexOut
    
    def estimateLeastSquaresEllipse(self,x,y):
        
        xMean = x.mean()
        yMean = y.mean()
        
        x = x-xMean
        y = y-yMean
                            
        mat = np.concatenate((x**2, x*y, y**2, x, y),axis=1)
        
        coeffs = self.LS.simpleLeastSquares(mat, np.ones_like(x))
        
        x0,y0,a,b,T = self.convertCoeffsToParams(coeffs[0],coeffs[1],coeffs[2],coeffs[3],coeffs[4])
        
        return x0+xMean, y0+yMean, a, b ,T
    
    #%% Estimate ellipse
        
    def estimateTotalLeastSquaresEllipse(self,x,y,Vxy,I):
        
        xOffset = np.mean(x)
        yOffset = np.mean(y)
                
        x = x-xOffset
        y = y-yOffset
                                    
        N = x.shape[0]*2
        
        i = np.arange(0,(N//2)*5,5)
        j = np.arange(0,N,2)
        
        J = np.zeros(((N//2)*5,N))
        J[i,j] = x.flatten()*2 
        J[i+1,j] = y.flatten() 
        J[i+3,j] = 1 
        J[i+1,j+1] = x.flatten()
        J[i+2,j+1] = y.flatten()*2
        J[i+4,j+1] = 1 
                
        Vmat = J@Vxy@J.T
        
        mat = np.concatenate((x**2, x*y, y**2, x, y),axis=1)
        
        coeffs, V = self.LS.indexedWeightedTotalLeastSquares(mat, np.ones_like(x), Vmat, np.zeros((x.shape[0], x.shape[0])), I)
        
        (x0,y0,a,b,T), Vell = self.defineEllipseFromCoeffs(coeffs, V)
        
        return x0+xOffset,y0+yOffset,a,b,T,Vell
    
    def convertCoeffsToParams(self,A,B,C,D,E):
        
        theta = 0.5*np.arctan2(-B, C-A)
        
        c = np.cos(theta)
        s = np.sin(theta)
        
        A1 = A*c**2 + B*c*s + C*s**2
        C1 = A*s**2 - B*c*s + C*c**2
        D1 = D*c + E*s
        E1 = -D*s + E*c
        F1 = 1 + (D1**2)/(4*A1) + (E1**2)/(4*C1)
        
        x0 = -c*D1/(2*A1) + s*E1/(2*C1)
        y0 = -s*D1/(2*A1) - c*E1/(2*C1)
        
        a = F1/A1
        b = F1/C1
        
        if a<0 or b<0: return [np.nan]*5
        else: return x0,y0,a**0.5,b**0.5,theta     
    
    def defineEllipseFromCoeffs(self,coeffs,V):
        
        A = coeffs[0,0]
        B = coeffs[1,0]
        C = coeffs[2,0]
        D = coeffs[3,0]
        E = coeffs[4,0]
                              
        J = self.JacobianFunction(A,B,C,D,E)
        
        Vell = J@V@J.T
        
        # asdasasd
        # test = corrMatrixfromCovMatrix(Vell)
        
        
        return self.convertCoeffsToParams(A,B,C,D,E), Vell
           
    def outsideIndex(self,x0,y0,a,b,theta,L=111):
                
        x = np.linspace(0, L-1, L).reshape(-1,1) - (L+1)//2
        y = np.linspace(0, L-1, L).reshape(1,-1) - (L+1)//2
        
        F = ( (x*np.cos(theta) - y*np.sin(theta)) / a)**2 + \
            ( (x*np.sin(theta) + y*np.cos(theta)) / b)**2 - \
            1
        
        O = np.empty((2,1), dtype = int)
        O[0,0] = int(np.round(x0))
        O[1,0] = int(np.round(y0))
        
        return np.vstack(np.nonzero(F>0)) + O - (L+1)//2
     
    def estimateTransform(self,C,P):
                       
        C1 = np.zeros((C.shape[0]*2, 6))
        C1[::2,0] = C[:,0]
        C1[::2,1] = C[:,1]
        C1[::2,2] = 1
        C1[1::2,3] = C[:,0]
        C1[1::2,4] = C[:,1]
        C1[1::2,5] = 1
        
        P1 = np.empty((P.shape[0]*2, 1))
        P1[::2,0] = P[:,0]
        P1[1::2,0] = P[:,1]
            
        W = np.ones((C1.shape[0],1))
        
        for i in range(3):
            W = W/W.max()
            A = np.linalg.inv(C1.T@(W*C1))@C1.T@(W*P1)
            W = 1/np.maximum(1e-4, np.sum((C1@A - P1)**2, axis=1, keepdims = True)**0.5)
                     
        V = np.linalg.inv(C1.T@(W*C1))
        
        A1 = np.empty((3,2))
        A1[:,0] = A[:3,0]
        A1[:,1] = A[3:,0]
                        
        return A1,V
    
    #%% Initialisation
    
    def initialiseNLS(self):
        
        numLines = self.options["numLines"]
        resLines = self.options["resLines"]
        
        def jacobianFunction(x,X):
    
            N = numLines
            M = resLines
            
            s2 = X[0]**2
            mu = np.repeat(X[1::2], M).reshape(-1,1)
            A = np.repeat(X[2::2], M).reshape(-1,1)
            
            J = np.zeros((M*N, N*2+1))
            
            H = np.exp(-(x - mu)**2/(2*s2))
            
            I = np.empty((N*M,2), dtype = int)
            I[:,0] = np.arange(0,M*N,1)
            I[:,1] = np.repeat(np.arange(1,2*N,2), M)
                
            J[:,0] = (A*(x - mu)**2*H/(s2**1.5)).flatten()
            J[I[:,0], I[:,1]] = (A*(x - mu)/s2*H).flatten()
            J[I[:,0], I[:,1]+1] = H.flatten()
            
            return J
    
        def transformFunction(x,X):
            
            M = resLines
            
            s2 = X[0]**2
            mu = np.repeat(X[1::2], M).reshape(-1,1)
            A = np.repeat(X[2::2], M).reshape(-1,1)
            
            out = A * np.exp(-(x-mu)**2/(2*s2))
               
            return out

        self.NLS = NonLinearSolver(jacobianFunction, transformFunction)
        self.NLS.setOptions(self.NLSoptions)
        
    def initialiseLS(self):
        
        self.LS = LinearSolver()
        self.LS.setOptions(self.LSoptions)
        
    #%% Troubleshooting
    
    def troubleshootLocalisation(self,blankImage,i):

        #Use cv2 to find board estimate
        t1=time.time()
        points = self.extractCameraPoints(blankImage)
        print("CV2: {0:.4g}".format(time.time()-t1))
        
        dot = points[i,:]
        
        t1=time.time()
        array,xAdd,yAdd = extractRegion(blankImage, dot, 50)
        
        gradient = sampleImageGradient(array)
        
        z,ix,iy = self.LSFsamples(gradient)
        
        x,y,Vxy,sigma,sigmaStd = self.refineLineSpreadFunctions(z,ix,iy)
        print("Edge refinement: {0:.4g}".format(time.time()-t1))

        t1=time.time()                           
        index = self.cleanEllipse(x,y).reshape(-1)

        xi = x[index]
        yi = y[index]
        print("Cleaning: {0:.4g}".format(time.time()-t1))
        
        #Iterate through
        t1=time.time()
        x0n,y0n,an,bn,Tn,Vout = self.estimateTotalLeastSquaresEllipse(x,y,Vxy,index)
        print("TLS est: {0:.4g}".format(time.time()-t1))
        
        xe,ye = self.closestPointOnEllipse(x,y,x0n,y0n,an,bn,Tn)

        plt.figure()
        plt.imshow(array.T, cmap = 'gray')
        plt.scatter(x[index],y[index],s=3,c='r')
        plt.scatter(x[np.invert(index)],y[np.invert(index)],s=3,c='b')
        for i in range(x.shape[0]):
            plotErrorEllipse(np.array([x[i,0],y[i,0]]), Vxy[2*i:2*i+2, 2*i:2*i+2])
        
        plt.figure()
        plt.imshow(array.T, cmap = 'gray')
        plt.scatter(xi,yi,s=3,c='r')
        plt.scatter(xe,ye,s=3,c='b')
        for i in range(xi.shape[0]):
            plt.plot([x[i,0],xe[i]], [y[i,0],ye[i]],'r')
        
        test = np.empty((x.shape[0]*2,1))
        test[::2,0] = x.flatten()-xe
        test[1::2,0] = y.flatten()-ye
        
        val = test * (np.linalg.pinv(Vxy) @ test)/100
        
        print("Position: {0:.4g},".format(x0n), " {0:.4g}".format(y0n))
        print("Covariance: {0:.4g},".format(Vout[0,0]**0.5), " {0:.4g},".format(Vout[1,1]**0.5), " {0:.4g}".format(Vout[0,1]/(Vout[0,0]**0.5*Vout[1,1]**0.5)))
        print("rChi2: {0:.04g}".format(np.sum(val)))
        
    def troubleshootArray(self,array):

        t1=time.time()       
        gradient = sampleImageGradient(array)
        
        z,ix,iy = self.LSFsamples(gradient)
        
        x,y,Vxy,sigma,sigmaStd = self.refineLineSpreadFunctions(z,ix,iy)
        print("Edge refinement: {}".format(time.time()-t1))

        t1=time.time()                           
        index = self.cleanEllipse(x,y).reshape(-1)

        xi = x[index]
        yi = y[index]
        print("Cleaning: {}".format(time.time()-t1))
        
        #Iterate through
        t1=time.time()
        x0n,y0n,an,bn,Tn,Vout = self.estimateTotalLeastSquaresEllipse(x,y,Vxy,index)
        print("TLS est: {}".format(time.time()-t1))
        
        xe,ye = self.closestPointOnEllipse(x,y,x0n,y0n,an,bn,Tn)

        plt.figure()
        plt.imshow(array.T, cmap = 'gray')
        plt.scatter(x[index],y[index],s=3,c='r')
        plt.scatter(x[np.invert(index)],y[np.invert(index)],s=3,c='b')
        
        plt.figure()
        plt.imshow(array.T, cmap = 'gray')
        plt.scatter(xi,yi,s=3,c='r')
        plt.scatter(xe,ye,s=3,c='b')
        for i in range(x.shape[0]):
            plt.plot([x[i,0],xe[i]], [y[i,0],ye[i]],'r')
        
        print("Position: {0:.4g}, {0:.4g}".format(x0n,y0n))
        print("Covariance: {0:.4g}, {0:.4g}, {0:.4g}" .format(Vout[0,0]**0.5, Vout[1,1]**0.5, Vout[0,1]/(Vout[0,0]*Vout[1,1])))
    
    def troubleshootInference(self,P,mappingX,mappingY,i):
        
        x0,y0,a,b,T = P[i,:]

        I = self.outsideIndex(x0,y0,a+5,b+5,T).T
        
        J = np.concatenate((mappingX[I[:,0], I[:,1]].reshape(-1,1), mappingY[I[:,0], I[:,1]].reshape(-1,1)), axis=1)
        
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(I[:,0], I[:,1], mappingX[I[:,0], I[:,1]].reshape(-1))
        
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(I[:,0], I[:,1], mappingY[I[:,0], I[:,1]].reshape(-1))
        
        A,VA = self.estimateTransform(I,J)
        
        x = np.concatenate((I, np.ones((I.shape[0], 1))),axis=1)
        
        test = x@A
        y = np.concatenate((mappingX[I[:,0], I[:,1]].reshape(-1,1), mappingY[I[:,0], I[:,1]].reshape(-1,1)), axis=1)
        
        error = test-y
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(I[:,0], I[:,1], error[:,0], s=1, c = error[:,0], cmap = 'seismic')
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(I[:,0], I[:,1], error[:,1], s=1, c = error[:,1], cmap = 'seismic')
            
    def plotEllipseFromMat(self,x0,y0,a,b,T,L=20,N=100):
        
        x,y = np.meshgrid(np.linspace(-L,L,N), np.linspace(-L,L,N), indexing = 'ij')
        
        vec = np.empty((3,N**2))
        vec[0,:] = x.flatten() + x0
        vec[1,:] = y.flatten() + y0
        vec[2,:] = 1
        
        M = self.defineMatrix(self.defineCoeffs(x0,y0,a,b,T))
    
        test = np.sum(vec * (M @ vec), axis=0)
        test = test.reshape(N,N)
    
        plt.imshow(np.abs(test<0).T, extent = [L+x0,-L+x0,-L+y0,L+y0], cmap = 'gray')
        
    def plotEllipseContour(self,x0,y0,a,b,T,L=20,N=100):
        
        x,y = np.meshgrid(np.linspace(x0-L,x0+L,N), np.linspace(y0-L,y0+L,N), indexing = 'ij')
        
        vec = np.empty((3,N**2))
        vec[0,:] = x.flatten()
        vec[1,:] = y.flatten()
        vec[2,:] = 1
        
        M = self.defineMatrix(self.defineCoeffs(x0,y0,a,b,T))
    
        test = np.sum(vec * (M @ vec), axis=0)
        test = test.reshape(N,N)
    
        plt.contour(x,y,test, [0], colors = 'red') 
    
    def defineMatrix(self,coeffs):
    
        A,B,C,D,E,F = coeffs    
    
        M = np.empty((3,3))
        M[0,0] = A
        M[0,1] = B/2
        M[0,2] = D/2
        M[1,0] = B/2
        M[1,1] = C
        M[1,2] = E/2
        M[2,0] = D/2
        M[2,1] = E/2
        M[2,2] = F
    
        return M

    def defineCoeffs(self,x,y,a,b,T):
        A = a**2*np.sin(T)**2 + b**2*np.cos(T)**2
        B = 2*(b**2 - a**2)*np.sin(T)*np.cos(T)
        C = a**2*np.cos(T)**2 + b**2*np.sin(T)**2
        D = -2*A*x - B*y
        E = -B*x - 2*C*y
        F = A*x**2 + B*x*y + C*y**2 - a**2*b**2
            
        return A,B,C,D,E,F
    
    def plotEllipseLine(self,x0,y0,a,b,T,N=100,c=''):
        
        theta = np.linspace(0,2*np.pi,N)
        
        x = a*np.cos(theta - T) + x0
        y = b*np.sin(theta - T) + y0
        
        plt.plot(x,y,c)
    
            
    