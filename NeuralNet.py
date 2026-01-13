import torch
import torch.nn as nn
try:
    from ConfigLoader import load_config
except ImportError:
    # Fallback if running standalone purely for testing without ConfigLoader in path 
    # though usually they are in same dir
    pass

class BoardEvaluator(nn.Module):
    def __init__(self, input_size=49, hidden_layers=None):
        super(BoardEvaluator, self).__init__()
        
        # Load config if hidden_layers not provided explicitly
        if hidden_layers is None:
            try:
                config = load_config()
                hidden_layers = config['model']['hidden_layers']
                input_size = config['model']['input_size']
            except:
                # Default fallback if config fails
                hidden_layers = [128, 64, 32]
                input_size = 49

        layers = []
        in_dim = input_size
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, 1)) # Output score
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
