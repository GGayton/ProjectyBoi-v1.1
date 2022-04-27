import sympy as sp
import os
import sys
import h5py
import matplotlib.pyplot as plt

from commonlib.console_outputs import ProgressBar

#%%

def defineEllipseJacobian():
    A,B,C,D,E = sp.symbols("A,B,C,D,E", real = True)
    
    
    theta = 0.5*sp.atan2(-B,C-A)
    
    c = sp.cos(theta)
    s = sp.sin(theta)
    
    A1 = A*c**2 + B*c*s + C*s**2
    C1 = A*s**2 - B*c*s + C*c**2
    D1 = D*c + E*s
    E1 = -D*s + E*c
    F1 = 1 + (D1**2)/(4*A1) + (E1**2)/(4*C1)
    
    x0 = -c*D1/(2*A1) + s*E1/(2*C1) 
    y0 = -s*D1/(2*A1) - c*E1/(2*C1)
    
    a = (F1/A1)**0.5
    b = (F1/C1)**0.5
    
    J = sp.Matrix([[0]*5]*5)
    
    bar = ProgressBar()
    bar.updateBar(0,26)
    
    v = [x0,y0,a,b,theta]
    w = [A,B,C,D,E]
    
    for i in range(5):
        for j in range(5):
            
            J[i,j] = v[i].diff(w[j])
            
            bar.updateBar(i*5+j+1, 26)
        

    f = sp.lambdify(w, J)
    bar.updateBar(26, 26)
    
    return f



#%%

if __name__ == "__main__":
    
    currentDir = os.getcwd()
    cutoff = currentDir.find("ProjectyBoy2000")
    assert cutoff != -1
    home = currentDir[:cutoff+16]
    
    if home not in sys.path:sys.path.append(home)
    
    A,B,C,D,E = sp.symbols("A,B,C,D,E", real = True)
    
    
    theta = 0.5*sp.atan2(-B,C-A)
    
    c = sp.cos(theta)
    s = sp.sin(theta)
    
    A1 = A*c**2 + B*c*s + C*s**2
    C1 = A*s**2 - B*c*s + C*c**2
    D1 = D*c + E*s
    E1 = -D*s + E*c
    F1 = 1 + (D1**2)/(4*A1) + (E1**2)/(4*C1)
    
    x0 = -c*D1/(2*A1) + s*E1/(2*C1) 
    y0 = -s*D1/(2*A1) - c*E1/(2*C1)
    
    a = (F1/A1)**0.5
    b = (F1/C1)**0.5
    
    J = sp.Matrix([[0]*5]*5)
    
    bar = ProgressBar()
    bar.updateBar(0,26)
    
    v = [x0,y0,a,b,theta]
    w = [A,B,C,D,E]
    
    for i in range(5):
        for j in range(5):
            
            J[i,j] = v[i].diff(w[j])
            
            bar.updateBar(i*5+j+1, 26)
        

    bar.updateBar(26, 26)