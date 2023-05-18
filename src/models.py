import copy
import functools
import logging
import math
import os
from tempfile import TemporaryDirectory
import time
from typing import Callable, List, Union
import numpy as np
import pandas as pd
from scipy.optimize import brent
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.functional import binary_cross_entropy, mse_loss


import torch
import torch.nn as nn
import numpy as np
import logging

from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data import random_split

from tqdm import tqdm

TRAIN_FRAC = 0.8
set_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLP2(nn.Module):
    def __init__(self, input_dim, num_classes, num_units=1, nonlin=nn.Softplus(), dropout_ratio=0.1):
        super().__init__()

        self.dense0 = nn.Linear(input_dim, num_units)

    def forward(self, X, **kwargs):
        X = self.dense0(X.float())
        # X = self.nonlin(X)
        # #X = self.dropout(X)
        # X = self.dense1(X).float()
        # X = self.nonlin(X)
        #X = self.output(X.float())
        return X


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, num_units=16, nonlin=nn.Softplus(), dropout_ratio=0.1):
        super().__init__()

        self.dense0 = nn.Linear(input_dim, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout_ratio)
        self.dense1 = nn.Linear(num_units, num_units)
        self.output = nn.Linear(num_units, num_classes)

    def forward(self, X, **kwargs):
        X = self.dense0(X.float())
        X = self.nonlin(X)
        #X = self.dropout(X)
        X = self.dense1(X).float()
        X = self.nonlin(X)
        X = self.output(X.float())
        return X

class Linear(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.dense0 = nn.Linear(input_dim, num_classes)
        
    def forward(self, X, **kwargs):
        return self.dense0(X)

 
class MLPRegressor():
    def __init__(self, input_dim=9, num_classes=2):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = MLP(input_dim, num_classes)
        self.criterion = nn.MSELoss()

    def update(self, config):
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.train_epochs = config["train_epochs"]
        self.model = MLP(self.input_dim, self.num_classes, num_units=config["hidden_size"], dropout_ratio=config["dropout_ratio"])
        #self.model = MLP2(self.input_dim, 1)
    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        predicted = outputs.data
        return predicted.detach().cpu().numpy().reshape(1, -1)[0]

    def score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        # predicted = torch.max(outputs.data, 1)
        # correct = (predicted == labels).sum().item()
        # total = y.shape[0]
        # return correct / total


    def fit(self, X, y, train_ratio=TRAIN_FRAC, device=set_device):
        self.device = device 

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)

        self.model = self.model.to(device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # criterion = nn.CrossEntropyLoss()
        
        trainset = TensorDataset(X, y)
        test_abs = int(len(trainset) * train_ratio)
        train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])
        train_subset = trainset
        trainloader = torch.utils.data.DataLoader(train_subset,batch_size=self.batch_size, shuffle=True, num_workers=8)
        valloader = torch.utils.data.DataLoader(val_subset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        for epoch in tqdm(range(0,self.train_epochs+1)):
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                outputs = outputs.to(torch.float64)
                loss = self.criterion(labels, outputs.squeeze(-1))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_steps += 1
                
            # print("[%d] loss: %.3f" % (epoch + 1,running_loss))
            
            
        total = 0
        error = 0
        self.model.eval()
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                predicted = outputs.data
                total += labels.size(0)
                error += torch.sum(torch.square(predicted - labels))

        # print("MSE", error / total)

        return error / total