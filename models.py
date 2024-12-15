import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


class FeedforwardNeuralNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, max_len):
        super(FeedforwardNeuralNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  
        self.fc1 = nn.Linear(embed_dim * max_len, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)  
        x = x.view(x.size(0), -1)  
        x = self.relu1(self.fc1(x))
        x = self.fc2(x).squeeze(-1)  
        return x

    
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, kernel_sizes, output_dim):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # Random embeddings
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # Add channel dimension for Conv2D
        conv_outs = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled_outs = [torch.max(c_out, dim=2)[0] for c_out in conv_outs]
        x = torch.cat(pooled_outs, dim=1)
        x = self.fc(self.dropout(x)).squeeze(-1)  # Output logits
        return x
    

class GatedRecurrentUnit(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, output_dim, dropout=0.5):
        super(GatedRecurrentUnit, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # Random embeddings
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, 
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Bidirectional GRU
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.gru(x)
        forward_hidden = hidden[-2, :, :]  # Last layer forward direction
        backward_hidden = hidden[-1, :, :]  # Last layer backward direction
        
        # Concatenate forward and backward hidden states
        combined_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)  # Shape: (batch_size, hidden_dim * 2)
        x = self.fc(self.dropout(combined_hidden)).squeeze(-1)  # Output logits
        return x