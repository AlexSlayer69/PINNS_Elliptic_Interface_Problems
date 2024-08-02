import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Based on https://github.com/1417573837/Deep-Learning-on-PDE/tree/main

"""

class VDL(nn.Module):
    def __init__(self, in_features, out_features, activation_functions,dropout_rate=0):
        super(VDL, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_functions = activation_functions
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        # Xavier initialization for weights
        nn.init.xavier_uniform_(self.weight)
        # Initialize biases to zero
        nn.init.zeros_(self.bias)
    
    def forward(self, input):
        # Perform linear transformation
        output = F.linear(input, self.weight, self.bias)
        
        # Apply separate activation functions
        activations = []
        for i in range(self.out_features):
            activation_fn = self.activation_functions[i % len(self.activation_functions)]
            
            activations.append(activation_fn(output[:,i]))
 
        return torch.stack(activations,dim = -1)
    
class VDLNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth, activation_functions):
        super(VDLNN, self).__init__()
        self.layers = []
        
        self.layers.append(VDL(input_size, hidden_size, activation_functions))
        for _ in range(depth - 2):
            self.layers.append(VDL(hidden_size, hidden_size, activation_functions))
        self.layers.append(VDL(hidden_size, output_size, activation_functions))
        
        self.model = nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.model(x)    