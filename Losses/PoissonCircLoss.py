import torch
import torch.nn as nn
import numpy as np
import math

class CircLoss():
    def __init__(self,n_points,int_pts,bound_pts,K):
        """
        n_points: number of points for domain
        int_pts = number of points per interface
        bound_ps = number of points per boundary
        K = [K_1,K_2]
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.iter = 1
        self.criterion = nn.MSELoss()
        
        self.n_points = n_points
        self.int_pts = int_pts
        self.bound_pts = bound_pts
        self.K = K
        
        x = torch.rand(self.n_points)*2 - 1
        y = torch.rand(self.n_points)*2 - 1
        self.W = torch.stack((x, y), dim=1).reshape(-1,2)
        xx,yy = self.W[:,0],self.W[:,1]
        mask1 = xx**2 + yy**2 <= 0.25
        mask2 = xx**2 + yy**2 >= 0.25
        self.W1 = self.W[mask1].to(self.device)
        self.W2 = self.W[mask2].to(self.device)
        
        angles = torch.linspace(0, 2 * torch.pi, self.int_pts) 
        x = 0.5 * torch.cos(angles)
        y = 0.5 * torch.sin(angles)
        self.G = torch.stack((x, y), dim=1).reshape(-1,2).to(self.device)
        
        x = torch.arange(0,1+(1/self.bound_pts),1/self.bound_pts)
        self.B1 = torch.stack((x,-torch.ones_like(x)),dim =1).reshape(-1,2).to(self.device)
        self.B2 = torch.stack((torch.ones_like(x),x),dim =1).reshape(-1,2).to(self.device)
        self.B3 = torch.stack((x,torch.ones_like(x)),dim =1).reshape(-1,2).to(self.device)
        self.B4 = torch.stack((-torch.ones_like(x),x),dim =1).reshape(-1,2).to(self.device)
        
        #All grads
        self.W1.requires_grad = True
        self.W2.requires_grad = True
        self.G.requires_grad = True

    def reset(self):
        self.iter = 1
    
        x = torch.rand(self.n_points)*2 - 1
        y = torch.rand(self.n_points)*2 - 1
        self.W = torch.stack((x, y), dim=1).reshape(-1,2)
        xx,yy = self.W[:,0],self.W[:,1]
        mask1 = xx**2 + yy**2 <= 0.25
        mask2 = xx**2 + yy**2 >= 0.25
        self.W1 = self.W[mask1].to(self.device)
        self.W2 = self.W[mask2].to(self.device)
        
        angles = torch.linspace(0, 2 * torch.pi, self.int_pts) 
        x = 0.5 * torch.cos(angles)
        y = 0.5 * torch.sin(angles)
        self.G = torch.stack((x, y), dim=1).reshape(-1,2).to(self.device)
        
        x = torch.arange(-1,1+(1/self.bound_pts),1/self.bound_pts)
        self.B1 = torch.stack((x,-torch.ones_like(x)),dim =1).reshape(-1,2).to(self.device)
        self.B2 = torch.stack((torch.ones_like(x),x),dim =1).reshape(-1,2).to(self.device)
        self.B3 = torch.stack((x,torch.ones_like(x)),dim =1).reshape(-1,2).to(self.device)
        self.B4 = torch.stack((-torch.ones_like(x),x),dim =1).reshape(-1,2).to(self.device)
        
        #All grads
        self.W1.requires_grad = True
        self.W2.requires_grad = True
        self.G.requires_grad = True
            
    def loss(self,model,mode,cond_func=None):
        
        if(mode == 'ipinn'):
            model.apply_activation = lambda x : cond_func(x,'0')    
            u1 = model(self.W1)
            g1 = model(self.G)
            
            model.apply_activation = lambda x : cond_func(x,'1')    
            u2 = model(self.W2)
            g2 = model(self.G)
            ub1 = model(self.B1)
            ub2 = model(self.B2)
            ub3 = model(self.B3)
            ub4 = model(self.B4)
            
        elif(mode == 'xpinn'):
            u1 = model[0](self.W1)
            g1 = model[0](self.G)  
            
            u2 = model[1](self.W2)
            g2 = model[1](self.G)
            ub1 = model[1](self.B1)
            ub2 = model[1](self.B2)
            ub3 = model[1](self.B3)
            ub4 = model[1](self.B4)
        
        # Interface Grads
        dG1_dW = torch.autograd.grad(
                    outputs=g1,
                    inputs=self.G,
                    grad_outputs=torch.ones_like(g1),
                    retain_graph=True,
                    create_graph=True
                )[0] 
        dG2_dW = torch.autograd.grad(
                    outputs=g2,
                    inputs=self.G,
                    grad_outputs=torch.ones_like(g2),
                    retain_graph=True,
                    create_graph=True
                )[0]
        
        f = self.G[:, 0]**2 + self.G[:, 1]**2 - 0.25
        # Compute the gradient of f with respect to the input points
        grad_f = torch.autograd.grad(
            outputs=f,
            inputs=self.G,
            grad_outputs=torch.ones_like(f),
            retain_graph=True,
            create_graph=True
        )[0]
        # Normalize the gradient to get the unit normal vector
        norm = torch.norm(grad_f, dim=1, keepdim=True)
        n2 = -grad_f / norm #from outside to inside
        # Calculate K \cdot (\nabla u)
        Ku1 = self.K[0] * dG1_dW
        Ku2 = self.K[1] * dG2_dW
        # Calculate (K \cdot (\nabla u)) \cdot n2
        result1 = torch.einsum('bi,bi->b', Ku1, n2)
        result2 = torch.einsum('bi,bi->b', Ku2, n2)
   
        # PDE grads
        du_dW1 = torch.autograd.grad(
            inputs=self.W1,
            outputs=u1,
            grad_outputs=torch.ones_like(u1),
            retain_graph=True,
            create_graph=True
        )[0]
        du_dxx1 = torch.autograd.grad(
            inputs=self.W1,
            outputs=du_dW1[:,0],
            grad_outputs=torch.ones_like(du_dW1[:,0]),
            retain_graph=True,
            create_graph=True
        )[0][:,0]
        du_dyy1 = torch.autograd.grad(
            inputs=self.W1,
            outputs=du_dW1[:,1],
            grad_outputs=torch.ones_like(du_dW1[:,1]),
            retain_graph=True,
            create_graph=True
        )[0][:,1]
        
        du_dW2 = torch.autograd.grad(
            inputs=self.W2,
            outputs=u2,
            grad_outputs=torch.ones_like(u2),
            retain_graph=True,
            create_graph=True
        )[0]
        du_dxx2 = torch.autograd.grad(
            inputs=self.W2,
            outputs=du_dW2[:,0],
            grad_outputs=torch.ones_like(du_dW2[:,0]),
            retain_graph=True,
            create_graph=True
        )[0][:,0]
        du_dyy2 = torch.autograd.grad(
            inputs=self.W2,
            outputs=du_dW2[:,1],
            grad_outputs=torch.ones_like(du_dW2[:,1]),
            retain_graph=True,
            create_graph=True
        )[0][:,1]
        
         
        loss_boundary = (self.criterion(ub1,torch.ones_like(ub1) + torch.log(2*torch.sqrt(self.B1[:,0]**2 + self.B1[:,1]**2).view_as(ub1))) +
                         self.criterion(ub2,torch.ones_like(ub2) + torch.log(2*torch.sqrt(self.B2[:,0]**2 + self.B2[:,1]**2).view_as(ub2))) +
                         self.criterion(ub3,torch.ones_like(ub3) + torch.log(2*torch.sqrt(self.B3[:,0]**2 + self.B3[:,1]**2).view_as(ub3))) +
                         self.criterion(ub4,torch.ones_like(ub4) + torch.log(2*torch.sqrt(self.B4[:,0]**2 + self.B4[:,1]**2).view_as(ub4))))
                        
        loss_deriv = self.criterion(result2-result1,-2*self.K[1]*torch.ones_like(result1))
        loss_jump = self.criterion(g1,g2)               
        loss_pde = self.criterion(du_dxx1,-du_dyy1) + self.criterion(du_dxx2,-du_dyy2)

        loss = loss_jump + loss_boundary + loss_deriv + loss_pde
        loss.backward()
        print('PDE: ',loss_pde.item(),'Bound: ',loss_boundary.item(),'jmp: ',loss_jump.item(),'deriv: ',loss_deriv.item())
        return loss     