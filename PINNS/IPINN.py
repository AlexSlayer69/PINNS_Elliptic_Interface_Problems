# libraries
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm 
import sys
import os 

#custom
sys.path.insert(0, os.path.abspath('./PINNS/models/'))
from IPINNLayer import IPINNLayer

class IPINN():
    def __init__(self,dimension,hidden_size,depth):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = IPINNLayer(input_size = dimension, hidden_size = hidden_size,output_size = 1,depth = depth).to(self.device)
        self.adam = torch.optim.Adam(self.model.parameters())
        self.loss_func = None

    def setLoss(self,loss_func):
        """
        loss_func : A function with two inputs:
        models: A list of models for each subdomain
        pts : list of all the training points
        """
        self.loss_func = lambda: loss_func(self.model)
        
    def train(self,iterations,clip_gradients = False):
        if(self.loss_func == None):
            print('Set the loss function for the model')
            return 
        
        self.model.train()
        if(clip_gradients):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
           
    def eval(self,X,activations):
        self.model.eval()      
        with torch.no_grad():
            X = X.to(self.device)
            outputs = []
            for i in range(len(activations)):
                self.model.apply_activation = activations[i]
                outputs.append(self.model(X))
        return outputs
        