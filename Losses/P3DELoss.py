#libraries
import torch
import torch.nn as nn
import numpy as np

#custom 
from Generator import generate_points_on_ellipsoid,generate_random_points_on_cube_surface

class P3DELoss:
    def __init__(self,batch_size,int_pts,bound_pts,K):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.MSELoss()
        self.iter = 1
        
        self.K = K
        self.batch_size = batch_size
        self.int_pts = int_pts
        self.bound_pts = bound_pts
        
        x = torch.rand(self.batch_size)*2 -1 
        y = torch.rand(self.batch_size)*2 -1
        z = torch.rand(self.batch_size)*2 -1
        self.W = torch.stack((x, y, z), dim=1)
        xx,yy,zz = self.W[:,0],self.W[:,1],self.W[:,2]
        mask = (2*xx**2 + 3*yy**2 + 6*zz**2 - 1.69 <= 0)
        self.W1 = self.W[mask]
        self.W2 = self.W[~mask]
        
        self.W1 = self.W1.to(self.device)
        self.W2 = self.W2.to(self.device)
        self.W1.requires_grad = True
        self.W2.requires_grad = True
        
        self.G = generate_points_on_ellipsoid(self.int_pts).to(self.device)
        self.G.requires_grad = True
        
        self.B1 = generate_random_points_on_cube_surface(bound_pts,[1,0,0]).to(self.device)
        self.B2 = generate_random_points_on_cube_surface(bound_pts,[-1,0,0]).to(self.device)
        self.B3 = generate_random_points_on_cube_surface(bound_pts,[0,1,0]).to(self.device)
        self.B4 = generate_random_points_on_cube_surface(bound_pts,[0,-1,0]).to(self.device)
        self.B5 = generate_random_points_on_cube_surface(bound_pts,[0,0,1]).to(self.device)
        self.B6 = generate_random_points_on_cube_surface(bound_pts,[0,0,-1]).to(self.device)
        
    def loss(self,model,cond_func):    
        self.iter += 1
        
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
        ub5 = model(self.B5)
        ub6 = model(self.B6)
        
        #PDE grads
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
        du_dzz1 = torch.autograd.grad(
            inputs=self.W1,
            outputs=du_dW1[:,2],
            grad_outputs=torch.ones_like(du_dW1[:,2]),
            retain_graph=True,
            create_graph=True
        )[0][:,2]
        
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
        du_dzz2 = torch.autograd.grad(
            inputs=self.W2,
            outputs=du_dW2[:,2],
            grad_outputs=torch.ones_like(du_dW2[:,2]),
            retain_graph=True,
            create_graph=True
        )[0][:,2]

        #deriv grads
        grad_g1 = torch.autograd.grad(
        outputs=g1,
        inputs=self.G,
        grad_outputs=torch.ones_like(g1),
        retain_graph=True,
        create_graph=True
        )[0]
        grad_g2= torch.autograd.grad(
            outputs=g2,
            inputs=self.G,
            grad_outputs=torch.ones_like(g2),
            retain_graph=True,
            create_graph=True
        )[0]
        
        f = 2 * self.G[:, 0]**2 + 3 * self.G[:, 1]**2 + 6 * self.G[:, 2]**2 - 1.69
        # Compute the gradient of f with respect to the input points
        grad_f = torch.autograd.grad(
            outputs=f,
            inputs=self.G,
            grad_outputs=torch.ones_like(f),
            retain_graph=True,
            create_graph=True
        )[0]
        # Normalize the gradient to get the unit normal vector
        #norm = torch.norm(grad_f, dim=1, keepdim=True)
        n2 = grad_f #/ norm #from outside to inside
        # Calculate K \cdot (\nabla u)
        Ku1 = self.K[0] * grad_g1
        Ku2 = self.K[1] * grad_g2
        # Calculate (K \cdot (\nabla u)) \cdot n2
        result1 = torch.einsum('bi,bi->b', Ku1, n2)
        result2 = torch.einsum('bi,bi->b', Ku2, n2)

        loss_boundary = (self.criterion(ub1,self.B1[:,0] + self.B1[:,1] + self.B1[:,2]) + 
                         self.criterion(ub2,self.B2[:,0] + self.B2[:,1] + self.B2[:,2]) +
                         self.criterion(ub3,self.B3[:,0] + self.B3[:,1] + self.B3[:,2]) + 
                         self.criterion(ub4,self.B4[:,0] + self.B4[:,1] + self.B4[:,2]) + 
                         self.criterion(ub5,self.B5[:,0] + self.B5[:,1] + self.B5[:,2]) + 
                         self.criterion(ub6,self.B6[:,0] + self.B6[:,1] + self.B6[:,2]))
    
        loss_jump = self.criterion(g2-g1,(self.G[:, 0] + self.G[:, 1] +self.G[:, 2]).view_as(g1) -
                                         (self.G[:, 0]**2 + self.G[:, 1]**2 + self.G[:, 2]**2).view_as(g1))
        
        loss_deriv = self.criterion(result2-result1,((240*self.G[:,0] - 16*self.G[:, 0]**2) + 
                                                     (360*self.G[:,1] - 24*self.G[:, 1]**2) + 
                                                     (720*self.G[:,2] - 48*self.G[:, 2]**2)).view_as(result1))
        
        loss_pde = (self.criterion(du_dxx1 + du_dyy1 + du_dzz1,(12/self.K[0])*torch.ones_like(du_dxx1)) + 
                    self.criterion(du_dxx2 + du_dyy2, -du_dzz2))
        
        loss = loss_jump + loss_boundary + (loss_deriv/60000) + (loss_pde/100)
        loss.backward()
        print('PDE: ',loss_pde.item()/10,'Bound: ',loss_boundary.item(),'jmp: ',loss_jump.item(),'deriv: ',loss_deriv.item()/60000)
        return loss

        

        
        
        
    
        
