import torch
import torch.nn as nn

'''
    Class for Neural Network models
    Supports RNN, LSTM, and BiLSTM architectures under a single class structure
'''

class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config["vocab_size"], config["embedding_dim"], padding_idx=0)
        self.arch = config["model"]
        self.num_layers = config["hidden_layers"]
        self.dropout = nn.Dropout(config["dropout"])
        self.hidden_size = config["hidden_size"]

        # set the model architecture based on config
        if self.arch == "RNN":
            self.rnn = nn.RNN(config["embedding_dim"], self.hidden_size, num_layers=self.num_layers, batch_first=True)
        elif self.arch == "LSTM":
            self.rnn = nn.LSTM(config["embedding_dim"], self.hidden_size, num_layers=self.num_layers, batch_first=True)
        elif self.arch == "BiLSTM":
            self.rnn = nn.LSTM(config["embedding_dim"], self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        else:
            raise ValueError("Unknown architecture")

        # fully connected layers with activation
        direction_factor = 2 if self.arch == "BiLSTM" else 1
        self.fc1 = nn.Linear(self.hidden_size * direction_factor, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 1)
        self.activation = getattr(torch, config["activation"]) if hasattr(torch, config["activation"]) else torch.tanh

    # forward pass for the network
    def forward(self, x):
        x = self.embed(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.activation(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze(-1)
