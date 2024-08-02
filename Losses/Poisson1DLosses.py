#libraries
import torch
import torch.nn as nn
import numpy as np

#custom 
import Generator

class Poisson1DLosses:
    """
    A class for all the losses of 1D Poisson Equation
    n_points: number of points to be used in training 
    sub_domains: number of sub-domains of the problem
    sampling: A bool that determines whether to use sampling or not
    h: It is the precision for Γ+ and Γ- 
    K : Values of K as given in the equation
    gamma : Points belonging to Γ
    """
    def __init__(self,n_points,sub_domains,K,gamma):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.iter = 1
        self.criterion = nn.MSELoss()
        
        self.n_points = n_points
        self.sub_domains = sub_domains
        self.gamma = gamma
        self.K = K
        
        self.h = 1/n_points
        self.W1 = torch.arange(gamma[0],gamma[1]+self.h,self.h).reshape(-1,1).to(self.device)
        self.W2 = torch.arange(gamma[1],gamma[2]+self.h,self.h).reshape(-1,1).to(self.device)
        self.W3 = torch.arange(gamma[2],gamma[3]+self.h,self.h).reshape(-1,1).to(self.device)
        self.W4 = torch.arange(gamma[3],gamma[4]+self.h,self.h).reshape(-1,1).to(self.device)
        self.W5 = torch.arange(gamma[4],gamma[5]+self.h,self.h).reshape(-1,1).to(self.device)
        
        self.B0 = torch.tensor([0.0]).reshape(-1,1).to(self.device)
        self.B1 = torch.tensor([1.0]).reshape(-1,1).to(self.device)
        
        self.G1 = torch.tensor([self.gamma[1]]).reshape(-1,1).to(self.device)
        self.G2 = torch.tensor([self.gamma[2]]).reshape(-1,1).to(self.device)
        self.G3 = torch.tensor([self.gamma[3]]).reshape(-1,1).to(self.device)
        self.G4 = torch.tensor([self.gamma[4]]).reshape(-1,1).to(self.device)
        
        
        self.W1.requires_grad = True
        self.W2.requires_grad = True
        self.W3.requires_grad = True
        self.W4.requires_grad = True
        self.W5.requires_grad = True
        self.G1.requires_grad = True 
        self.G2.requires_grad = True 
        self.G3.requires_grad = True 
        self.G4.requires_grad = True 

    def reset(self):
        self.iter = 1      
        self.W1 = torch.arange(self.gamma[0],self.gamma[1]+self.h,self.h).reshape(-1,1).to(self.device)
        self.W2 = torch.arange(self.gamma[1],self.gamma[2]+self.h,self.h).reshape(-1,1).to(self.device)
        self.W3 = torch.arange(self.gamma[2],self.gamma[3]+self.h,self.h).reshape(-1,1).to(self.device)
        self.W4 = torch.arange(self.gamma[3],self.gamma[4]+self.h,self.h).reshape(-1,1).to(self.device)
        self.W5 = torch.arange(self.gamma[4],self.gamma[5]+self.h,self.h).reshape(-1,1).to(self.device)
        
        self.B0 = torch.tensor([0.0]).reshape(-1,1).to(self.device)
        self.B1 = torch.tensor([1.0]).reshape(-1,1).to(self.device)
        
        self.G1 = torch.tensor([self.gamma[1]]).reshape(-1,1).to(self.device)
        self.G2 = torch.tensor([self.gamma[2]]).reshape(-1,1).to(self.device)
        self.G3 = torch.tensor([self.gamma[3]]).reshape(-1,1).to(self.device)
        self.G4 = torch.tensor([self.gamma[4]]).reshape(-1,1).to(self.device)
        
        self.W1.requires_grad = True
        self.W2.requires_grad = True
        self.W3.requires_grad = True
        self.W4.requires_grad = True
        self.W5.requires_grad = True
        self.G1.requires_grad = True 
        self.G2.requires_grad = True 
        self.G3.requires_grad = True 
        self.G4.requires_grad = True 
            
    def loss(self,model,mode,cond_func=None):
        """
        Loss function for a traditional PINN 
        model:IT is the NN model(s) that is to be trained 
        mode: Which NN to train
        """        
        self.iter += 1
        
        if(mode == 'pinn'):
            u1 = model(self.W1)
            u2 = model(self.W2)
            u3 = model(self.W3)
            u4 = model(self.W4)
            u5 = model(self.W5)
            
            ub0 = model(self.B0)
            ub1 = model(self.B1)
            
            g1 = model(self.G1)
            g2 = model(self.G2)
            g3 = model(self.G3)
            g4 = model(self.G4)
            g_1 = g1
            g_2 = g2
            g_3 = g3
            g_4 = g4
        elif(mode == 'xpinn'):
            u1 = model[0](self.W1)
            u2 = model[1](self.W2)
            u3 = model[2](self.W3)
            u4 = model[3](self.W4)
            u5 = model[4](self.W5)
            
            ub0 = model[0](self.B0)
            ub1 = model[4](self.B1)
            
            g1 = model[0](self.G1)
            g2 = model[1](self.G2)
            g3 = model[2](self.G3)
            g4 = model[3](self.G4)
            
            g_1 = model[1](self.G1)
            g_2 = model[2](self.G2)
            g_3 = model[3](self.G3)
            g_4 = model[4](self.G4)
        elif(mode == 'ipinn'):    
            model.apply_activation = lambda x : cond_func(x,'0')
            u1 = model(self.W1)
            ub0 = model(self.B0)
            g1 = model(self.G1)
            model.apply_activation = lambda x : cond_func(x,'1')
            u2 = model(self.W2)
            g_1 = model(self.G1)
            g2 = model(self.G2)
            model.apply_activation = lambda x : cond_func(x,'2')
            u3 = model(self.W3)
            g_2 = model(self.G2)
            g3 = model(self.G3)
            model.apply_activation = lambda x : cond_func(x,'3')
            u4 = model(self.W4)
            g_3 = model(self.G3)
            g4 = model(self.G4)
            model.apply_activation = lambda x : cond_func(x,'4')
            u5 = model(self.W5)
            g_4 = model(self.G4)
            ub1 = model(self.B1)
                
        du_dx1 = torch.autograd.grad(
                        inputs = self.W1,
                        outputs = u1,
                        grad_outputs = torch.ones_like(u1),
                        retain_graph = True,
                        create_graph = True
                        )[0]
        du_dxx1 = torch.autograd.grad(
                        inputs = self.W1,
                        outputs = du_dx1,
                        grad_outputs = torch.ones_like(du_dx1),
                        retain_graph = True,
                        create_graph = True
                        )[0]
        
        du_dx2 = torch.autograd.grad(
                        inputs = self.W2,
                        outputs = u2,
                        grad_outputs = torch.ones_like(u2),
                        retain_graph = True,
                        create_graph = True
                        )[0]
        du_dxx2 = torch.autograd.grad(
                        inputs = self.W2,
                        outputs = du_dx2,
                        grad_outputs = torch.ones_like(du_dx2),
                        retain_graph = True,
                        create_graph = True
                        )[0]
        
        du_dx3 = torch.autograd.grad(
                        inputs = self.W3,
                        outputs = u3,
                        grad_outputs = torch.ones_like(u3),
                        retain_graph = True,
                        create_graph = True
                        )[0]
        du_dxx3 = torch.autograd.grad(
                        inputs = self.W3,
                        outputs = du_dx3,
                        grad_outputs = torch.ones_like(du_dx3),
                        retain_graph = True,
                        create_graph = True
                        )[0]

        du_dx4 = torch.autograd.grad(
                        inputs = self.W4,
                        outputs = u4,
                        grad_outputs = torch.ones_like(u4),
                        retain_graph = True,
                        create_graph = True
                        )[0]
        du_dxx4 = torch.autograd.grad(
                        inputs = self.W4,
                        outputs = du_dx4,
                        grad_outputs = torch.ones_like(du_dx4),
                        retain_graph = True,
                        create_graph = True
                        )[0]
            
        du_dx5 = torch.autograd.grad(
                        inputs = self.W5,
                        outputs = u5,
                        grad_outputs = torch.ones_like(u5),
                        retain_graph = True,
                        create_graph = True
                        )[0]
        du_dxx5 = torch.autograd.grad(
                        inputs = self.W5,
                        outputs = du_dx5,
                        grad_outputs = torch.ones_like(du_dx5),
                        retain_graph = True,
                        create_graph = True
                        )[0] 
        
        dg_dx1 = torch.autograd.grad(
                        inputs = self.G1,
                        outputs = g1,
                        grad_outputs = torch.ones_like(g1),
                        retain_graph = True,
                        create_graph = True
                        )[0] 
        dgg_dx1 = torch.autograd.grad(
                        inputs = self.G1,
                        outputs = g_1,
                        grad_outputs = torch.ones_like(g_1),
                        retain_graph = True,
                        create_graph = True
                        )[0] 
        
        dg_dx2 = torch.autograd.grad(
                        inputs = self.G2,
                        outputs = g2,
                        grad_outputs = torch.ones_like(g2),
                        retain_graph = True,
                        create_graph = True
                        )[0] 
        dgg_dx2 = torch.autograd.grad(
                        inputs = self.G2,
                        outputs = g_2,
                        grad_outputs = torch.ones_like(g_2),
                        retain_graph = True,
                        create_graph = True
                        )[0] 
        
        dg_dx3 = torch.autograd.grad(
                        inputs = self.G3,
                        outputs = g3,
                        grad_outputs = torch.ones_like(g3),
                        retain_graph = True,
                        create_graph = True
                        )[0] 
        dgg_dx3 = torch.autograd.grad(
                        inputs = self.G3,
                        outputs = g_3,
                        grad_outputs = torch.ones_like(g_3),
                        retain_graph = True,
                        create_graph = True
                        )[0] 
        
        dg_dx4 = torch.autograd.grad(
                        inputs = self.G4,
                        outputs = g4,
                        grad_outputs = torch.ones_like(g4),
                        retain_graph = True,
                        create_graph = True
                        )[0] 
        dgg_dx4 = torch.autograd.grad(
                        inputs = self.G4,
                        outputs = g_4,
                        grad_outputs = torch.ones_like(g_4),
                        retain_graph = True,
                        create_graph = True
                        )[0]
       
        loss_pde = (self.criterion(self.K[0]*du_dxx1,-torch.ones_like(du_dxx1)) + 
                    self.criterion(self.K[1]*du_dxx2,-torch.ones_like(du_dxx2)) +  
                    self.criterion(self.K[2]*du_dxx3,-torch.ones_like(du_dxx3)) + 
                    self.criterion(self.K[3]*du_dxx4,-torch.ones_like(du_dxx4)) + 
                    self.criterion(self.K[4]*du_dxx5,-torch.ones_like(du_dxx5)))
        
        loss_boundary = self.criterion(ub0,torch.zeros_like(ub0)) + self.criterion(ub1,torch.zeros_like(ub1)) 
       
        loss_jump = (self.criterion(g1,g_1) + self.criterion(g2,g_2) + 
                     self.criterion(g3,g_3) + self.criterion(g4,g_4))
        
        loss_deriv = (self.criterion(self.K[0]*dg_dx1,self.K[1]*dgg_dx1) + 
                      self.criterion(self.K[1]*dg_dx2,self.K[2]*dgg_dx2) + 
                      self.criterion(self.K[2]*dg_dx3,self.K[3]*dgg_dx3) + 
                      self.criterion(self.K[3]*dg_dx4,self.K[4]*dgg_dx4))
        
        loss = loss_pde + loss_boundary + loss_deriv + loss_jump
        #print('lp',loss_pde.item(),'lb',loss_boundary.item(),'lj',loss_jump.item(),'ld',loss_deriv.item())
        loss.backward()
        return loss