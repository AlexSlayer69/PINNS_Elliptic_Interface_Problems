import math
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol
import numpy as np

class Poisson1D():
    """
    The Exact solution for One dimensional Poisson Equation with dirchlet boundary conditions
    ∇(K_m ∇u) = -1 in Ω_m
    [[u]] = 0 in Γ_int
    [[K∇u]] = 0 in Γ_int
    u(0) = u(1) = 0    
    """
    def __init__(self,K,gamma):
        self.K = K
        self.gamma = gamma
        
        inv_K = []
        for i in range(0,len(self.K)-1):
            inv_K.append((1.0/self.K[i]) - (1.0/self.K[i+1]))
        
        # Define the symbol and the inv_K and K arrays
        b_0 = Symbol('b')
        expression = 0

        b_0 = Symbol('b')
        l = (b_0*self.gamma[4] - ((self.gamma[4]**2)/2))*inv_K[3]
        self.B = solve(l+ (b_0*self.gamma[3] - ((self.gamma[3]**2)/2))*inv_K[2] + (b_0*self.gamma[2] - ((self.gamma[2]**2)/2))*inv_K[1] + (b_0*self.gamma[1] - ((self.gamma[1]**2)/2))*inv_K[0] + ((b_0 - 0.5)/K[4]), b_0)[0]
        
        self.C = [0,]
        for i in range(1,len(K)):
            self.C.append((self.B*self.gamma[i] - ((self.gamma[i]**2)/2))*inv_K[i-1] + self.C[i-1])
            
        self.equation = np.vectorize(self.solution)
        
    def solution(self,x):
        """
            m (int) is the mth sub-domain
            x (float) {0<=x<=1} 
            solution is of type ax^2 + bx + c
        """
        m = math.floor(x*len(self.K))
        m = np.clip(m,0,len(self.K) - 1)
        a = -1.0/(2*self.K[m])
        b = self.B/self.K[m]
        return (a*(x**2) + b*x + self.C[m])       
    
    def plot(self,n_points):
        xs = np.linspace(0,1,n_points)
        ys = []
        for x in xs: 
            ys.append(self.equation(x))
        plt.figure(figsize=(5,4))    
        plt.plot(xs,ys)
        plt.show()
    