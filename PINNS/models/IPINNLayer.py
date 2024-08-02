import torch
import torch.nn as nn

"""
Creates an IPINN Layer as described in : http://dx.doi.org/10.2139/ssrn.4766623

Citation:
Sarma, Antareep and Roy, Sumanta and Annavarapu, 
Chandrasekhar and Roy, Pratanu and Jagannathan, Sriram, 
Interface Pinns: A Framework of Physics-Informed Neural Networks for Interface Problems. 
Available at SSRN: https://ssrn.com/abstract=4766623 or http://dx.doi.org/10.2139/ssrn.4766623

""" 

class IPINNLayer(nn.Module):
    def __init__(self, input_size, hidden_size ,output_size, depth, apply_activation=None):
        super(IPINNLayer, self).__init__()
        self.layers = nn.ModuleList()
        
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        for _ in range(depth):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # If no external activation function is provided, use the internal one
        if apply_activation is None:
            self.apply_activation = self.default_apply_activation
        else:
            self.apply_activation = apply_activation
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.apply_activation(x)
        
        for layer in self.layers:
            x = layer(x)
            x = self.apply_activation(x)
        
        x = self.output_layer(x)
        return x
    
    def reset_parameters(self):
        # Xavier initialization for weights
        nn.init.xavier_uniform_(self.weight)
        # Initialize biases to zero
        nn.init.zeros_(self.bias)
    
    def default_apply_activation(self):
            pass