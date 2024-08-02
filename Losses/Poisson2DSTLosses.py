import torch
import torch.nn as nn

#custom
import Generator

class P2DSTLoss:
    def __init__(self,n_points,int_pts,bound_pts,K):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.MSELoss()
        self.iter = 1
        
        self.K = K
        self.n_points = n_points
        self.int_pts = int_pts
        self.bound_pts = bound_pts
        
        x = torch.rand(n_points) 
        y = torch.rand(n_points)
        self.W = torch.stack((x, y), dim=1)
        xx,yy = self.W[:,0],self.W[:,1]
        mask = (yy <= xx)
        self.W1 = self.W[mask]
        self.W2 = self.W[~mask]
        
        self.W1 = self.W1.to(self.device)
        self.W2 = self.W2.to(self.device)
        self.W1.requires_grad = True
        self.W2.requires_grad = True
        

        x = torch.arange(0,1+(1/int_pts),1/int_pts)
        self.G = torch.stack((x,x),dim = 1).to(self.device)
        self.G.requires_grad = True
        
        x = torch.arange(0,1+(1/bound_pts),1/bound_pts)
        
        self.bx0 = torch.stack((x,torch.zeros_like(x)),dim = 1).to(self.device)
        self.b0y = torch.stack((torch.zeros_like(x),x),dim = 1).to(self.device)
        
        self.bx1 = torch.stack((x,torch.ones_like(x)),dim = 1).to(self.device)
        self.b1y = torch.stack((torch.ones_like(x),x),dim = 1).to(self.device)
    
    def reset(self):
        self.iter = 1
        x = torch.rand(self.n_points) 
        y = torch.rand(self.n_points)
        self.W = torch.stack((x, y), dim=1)
        xx,yy = self.W[:,0],self.W[:,1]
        mask = (yy <= xx)
        self.W1 = self.W[mask]
        self.W2 = self.W[~mask]
        
        self.W1 = self.W1.to(self.device)
        self.W2 = self.W2.to(self.device)
        self.W1.requires_grad = True
        self.W2.requires_grad = True
        

        x = torch.arange(0,1+(1/self.int_pts),1/self.int_pts)
        self.G = torch.stack((x,x),dim = 1).to(self.device)
        self.G.requires_grad = True
        
        x = torch.arange(0,1+(1/self.bound_pts),1/self.bound_pts)
        
        self.bx0 = torch.stack((x,torch.zeros_like(x)),dim = 1).to(self.device)
        self.b0y = torch.stack((torch.zeros_like(x),x),dim = 1).to(self.device)
        
        self.bx1 = torch.stack((x,torch.ones_like(x)),dim = 1).to(self.device)
        self.b1y = torch.stack((torch.ones_like(x),x),dim = 1).to(self.device)
        
        
    def loss(self,model,mode,cond_func=None):
        self.iter += 1
        
        #sampling code here

        if(mode == 'ipinn'):
            model.apply_activation = lambda x : cond_func(x,'0')
            u1 = model(self.W1)
            ubx0 = model(self.bx0)
            ub1y = model(self.b1y)
            g1 = model(self.G)
        
            model.apply_activation = lambda x : cond_func(x,'1')
            u2 = model(self.W2)
            ubx1 = model(self.bx1)
            ub0y = model(self.b0y)   
            g2 = model(self.G) 
        elif(mode == 'xpinn'):
            u1 = model[0](self.W1)
            ubx0 = model[0](self.bx0)
            ub1y = model[0](self.b1y)
            g1 = model[0](self.G)    
            
            u2 = model[1](self.W2)
            ubx1 = model[1](self.bx1)
            ub0y = model[1](self.b0y)   
            g2 = model[1](self.G) 
        
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
        
        grad_u1_interface = torch.autograd.grad(
            outputs=g1,
            inputs=self.G,
            grad_outputs=torch.ones_like(g1),
            retain_graph=True,
            create_graph=True
        )[0]

        grad_u2_interface = torch.autograd.grad(
            outputs=g2,
            inputs=self.G,
            grad_outputs=torch.ones_like(g2),
            retain_graph=True,
            create_graph=True
        )[0]

        normal_derivative_u1 = self.K[0]*(grad_u1_interface[:, 0] - grad_u1_interface[:, 1])
        normal_derivative_u2 = self.K[1]*(grad_u2_interface[:, 0] - grad_u2_interface[:, 1])
        
        
        loss_boundary = (self.criterion(ubx0, (self.bx0[:, 0]**2).view_as(ubx0)) + 
                         self.criterion(ub0y, (self.b0y[:,1]**2).view_as(ub0y)) + 
                         self.criterion(ubx1, (self.bx1[:, 0]**2).view_as(ubx1) + torch.ones_like(ubx1)) + 
                         self.criterion(ub1y, (self.b1y[:, 1]).view_as(ub1y) + torch.ones_like(ub1y)))
                        
        loss_jump = self.criterion(g1,g2)
        loss_deriv = self.criterion(normal_derivative_u2 - normal_derivative_u1,-self.G[:, 1])
        
        loss_pde = (self.criterion(self.K[0] * (du_dxx1 + du_dyy1),torch.ones_like(du_dxx1)) +
                    self.criterion(self.K[1] * (du_dxx2 + du_dyy2),torch.ones_like(du_dxx2)))

        loss = loss_jump + loss_boundary + loss_deriv + loss_pde
        loss.backward()
        print('PDE: ',loss_pde.item(),'Bound: ',loss_boundary.item(),'jmp: ',loss_jump.item(),'deriv: ',loss_deriv.item())
        return loss
    