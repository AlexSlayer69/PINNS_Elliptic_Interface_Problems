import numpy as np

class Elliptical:
    def __init__(self):
        
        self.equation = np.vectorize(self.solution)
    
    def solution(self,x,y,z):
        if(2*x**2 + 3*y**2 + 6*z**2 - 1.69 <= 0):
            return x**2 + y**2 + z**2
        else:
            return x + y + z