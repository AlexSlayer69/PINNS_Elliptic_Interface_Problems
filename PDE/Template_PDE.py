from abc import ABC, abstractmethod
import numpy as np

class AbstractPDE(ABC):
    
    def __init__(self):
        """
        Initializes the AbstractPDE class.
        Sets up a vectorized version of the solution method.
        """
        self.equation = np.vectorize(self.solution)
        pass
    
    @abstractmethod
    def solution(self, *args):
        """
        Abstract method to compute the solution of the differential equation at a given point in space.
        
        Parameters:
        - *args (floats): The position in the domain in each direction (x, y, z, etc.).

        Returns:
        - float: The value of the solution at x.
        """
        pass
    