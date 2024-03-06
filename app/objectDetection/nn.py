import torch
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self, ind=2, w=64, w2=64, outd=1, lr=0.1):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(ind, w)
        self.fc2 = nn.Linear(w, w2)
        self.fc3 = nn.Linear(w2, outd)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

    def train(self, x, t):
        self.optimizer.zero_grad()
        output = self.forward(x)
        loss = self.criterion(output, t)
        loss.backward()
        self.optimizer.step()
