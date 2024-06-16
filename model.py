import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):

    def __init__(self, input_dim, fc_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.fc3 = nn.Linear(fc_dim, output_dim)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        output = self.fc1(x)
        output = self.leaky_relu(output)
        output = self.fc2(output)
        output = self.leaky_relu(output)
        output = self.fc3(output)

        return output