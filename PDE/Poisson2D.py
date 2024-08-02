import numpy as np
import math

class Poisson2DST:
    """
    The Exact solution for Two dimensional Poisson Equation with dirchlet boundary conditions
    ∇(K_m ∇u) = 1 in Ω_m
    [[u]] = 0 in Γ_int
    [[K∇u]].n_2 = -y/sqrt(2) in Γ_int
    u_m = (Λ_m)^d    
    """
    def __init__(self):
        self.equation = np.vectorize(self.solution)

    def solution(self,x, y):
        if(y <= x):
            return x**2 + x*y
        else:
            return x**2 + y**2 
        
class Poisson2DCirc:
    def __init__(self):
        self.equation = np.vectorize(self.solution)

    def solution(self,x, y):
        if(x**2 + y**2 <= 0.25):
            return 1
        else:
            return 1 + math.log(2*math.sqrt(x**2 + y**2))        