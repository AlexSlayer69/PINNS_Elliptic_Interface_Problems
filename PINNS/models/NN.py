import torch
import torch.nn as nn

class NN(nn.Module):
    """
    Defines a fully connected Neural Network Model
    """
    def __init__(self,input_size,hidden_size,output_size,depth,act = nn.Tanh):
        super(NN,self).__init__()

        self.layers = nn.ModuleList()
        self.input_layer = nn.Linear(input_size, hidden_size)
        for _ in range(depth):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        self.act = act()
        
    def reset_parameters(self):
        # Xavier initialization for weights
        nn.init.xavier_uniform_(self.weight)
        # Initialize biases to zero
        nn.init.zeros_(self.bias)    
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.act(x)
        
        for layer in self.layers:
            x = layer(x)
            x = self.act(x)
        
        x = self.output_layer(x)
        return x    