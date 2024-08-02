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

class XPINN:
    """
    Creates a PINN architecture based on XPINNs as described in: https://doi.org/10.4208/cicp.OA-2020-0164
    Github Link : https://github.com/AmeyaJagtap/XPINNs?tab=readme-ov-file
    @article{jagtap2020extended,
    title={Extended physics-informed neural networks (xpinns): A generalized space-time domain decomposition based deep learning framework for nonlinear         partial differential equations},
    author={Jagtap, Ameya D and Karniadakis, George Em},
    journal={Communications in Computational Physics},
    volume={28},
    number={5},
    pages={2002--2041},
    year={2020}
    }
    """
    def __init__(self,num_domains,dimension,hidden_size,depth,act = nn.Tanh):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.models = []
        self.optimizers = []
        #self.lbfgss = []
        self.num_domains = num_domains
        
        for i in range(self.num_domains):
            # One model for each sub-domain
            model = NN(input_size=dimension,hidden_size=hidden_size,output_size=1,depth=depth,act=act).to(self.device)
            self.models.append(model)
            self.optimizers.append(torch.optim.Adam(self.models[i].parameters(),lr = 0.002))
            # self.lbfgss.append(
            #     torch.optim.LBFGS(
            #         self.models[i].parameters(), 
            #         lr=1.0, 
            #         max_iter=50000, 
            #         max_eval=50000, 
            #         history_size=50,
            #         tolerance_grad=1e-10, 
            #         tolerance_change=1.0 * np.finfo(float).eps,
            #         line_search_fn="strong_wolfe",   # better numerical stability
            #     ) 
            # )

        self.loss_func = None
        
    def setLoss(self,loss_func):
        """
        loss_func : A function with two inputs:
        models: A list of models for each subdomain
        pts : list of all the training points
        """
        self.loss_func = lambda: loss_func(self.models)

    def train(self,iterations,clip_gradients = False):
        if(self.loss_func == None):
            print('Set the loss function for the model')
            return 

        for i in range(self.num_domains):
            self.models[i].train()
            
        if(clip_gradients):
            for i in range(self.num_domains):   
                torch.nn.utils.clip_grad_norm_(self.models[i].parameters(), 1.0)
            
        err = []
        
        for epoch in tqdm(range(iterations)):
            if(epoch % 100 == 0):
                ls = self.loss_func().cpu().detach().numpy()
                print('Epoch: ',epoch,' Loss: ',ls)
                err.append(ls)
            
            for i in range(self.num_domains):    
                self.optimizers[i].zero_grad()    
                self.optimizers[i].step(self.loss_func)
         
        # for _ in tqdm(range(iterations2)):
        #     self.lbfgss[i].zero_grad() 
        #     self.lbfgss[i].step(self.loss_func)     
        #     ls = self.loss_func().cpu().detach().numpy()
        #     print('Epoch: ',epoch,' Loss: ',ls)
        #     err.append(ls)
    
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
    
    