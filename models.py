import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes, num_filters):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        conv_outs = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled_outs = [F.max_pool1d(c_out, c_out.size(2)).squeeze(2) for c_out in conv_outs]
        x = torch.cat(pooled_outs, dim=1)
        x = self.fc(x)
        return x

class FeedforwardNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class GRUNeuralNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=1, bidirectional=False, dropout=0.3):
        super(GRUNeuralNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Embedding
        x = self.embedding(x)
        # GRU
        gru_out, _ = self.gru(x)
        # Use the last hidden state for classification
        x = gru_out[:, -1, :]
        x = self.dropout(x)
        # Fully connected layer
        x = self.fc(x)
        return x
