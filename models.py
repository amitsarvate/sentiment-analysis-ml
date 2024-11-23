import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class FeedforwardNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()


        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu1 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        
        out = self.fc2(out)
        out = self.relu2(out)

        out = self.fc3(out)

        return F.softmax(out, dim=1)