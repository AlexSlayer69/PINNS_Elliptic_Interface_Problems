#libraries
import torch 
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sys
import os
#custom
sys.path.insert(0, os.path.abspath('./PINNS/models/'))
from VDL import VDLNN

"""
A custom NN architecture which uses VDLNN models in an XPINN framework
"""

class VDLNet:
    def __init__(self,num_domains,dimension,activations,hidden_size,depth):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.models = []
        self.optimizers = []
        self.num_domains = num_domains
        for i in range(self.num_domains):
            model = VDLNN(input_size=dimension,hidden_size=hidden_size,output_size=1,depth=depth,activation_functions=activations,).to(self.device)
            self.models.append(model)
            self.optimizers.append(torch.optim.Adam(self.models[i].parameters()))
        
        self.loss_func = None
        

    def setLoss(self,loss_func):
        """
        loss_func : A function with two inputs:
        models: A list of models for each subdomain
        """
        self.loss_func = lambda: loss_func(self.models)

    def train(self,iterations,resample = 1):
        for i in range(self.num_domains):
            self.models[i].train()
        
        err = []
        for epoch in tqdm(range(iterations)):
            if(epoch % 100 == 0):
                ls = self.loss_func().cpu().detach().numpy()
                print('Epoch: ',epoch,' Loss: ',ls)
                err.append(ls)
                
            for i in range(self.num_domains):    
                self.optimizers[i].zero_grad()    
                self.optimizers[i].step(self.loss_func)
            
        plt.plot(np.log(err))
        plt.title('Loss')
        plt.show()

    def eval(self,X):
        for i in range(self.num_domains):
            self.models[i].eval()
        with torch.no_grad():
            X = X.to(self.device)
            outputs = []
            for i in range(self.num_domains):
                outputs.append(self.models[i](X))
        return outputs       
    
             