#libraries
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sys
import os

#custom
sys.path.insert(0, os.path.abspath('./PINNS/models/'))
from NN import NN

class PINN1D:
    """
    An Unmodified PINN network
    PDE: A PDE object as defined in PDE folder
    n_points: Number of trainingp points
    """
    def __init__(self,hidden_size,depth,act = nn.Tanh):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = NN(
            input_size=1,
            hidden_size=hidden_size,
            output_size=1,
            depth=depth,
            act=act
        ).to(self.device)
        
        self.adam = torch.optim.Adam(self.model.parameters())
        
        self.loss_func = None
        
    def setLoss(self,loss_func):
        """
        loss_func : A function with two inputs:
        model: The NN in use
        """
        self.loss_func = lambda : loss_func(self.model)    
    
    def train(self,iterations):
        if(self.loss_func == None):
            print('Set the loss function for the model')
            return 
        self.model.train()
        
        err = []
        for epoch in tqdm(range(iterations)):    
            if(epoch % 100 == 0):
                ls = self.loss_func().cpu().detach().numpy()
                print('Epoch: ',epoch,' Loss: ',ls)
                err.append(ls)
                
            self.adam.zero_grad()  
            self.adam.step(self.loss_func)  
             
        plt.plot(np.log(err))
        plt.title('Loss')
        plt.show()        
            
    def eval(self,X):
        """
        X : Is a torch tensor of appropriate dimension
        """
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X).reshape(-1,1).cpu().numpy()     
        return y_pred      
    