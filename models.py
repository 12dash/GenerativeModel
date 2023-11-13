import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, output_dim)
        
        self.activation = F.relu
        
    def forward(self, z):
        x = F.relu(self.fc1(z))
        if self.training:
            x = x+torch.randn((x.size(0),x.size(1)))/10
        x = self.activation(self.fc2(x))
        if self.training:
            x = x+torch.randn((x.size(0),x.size(1)))/10
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

        self.activation = F.relu
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x