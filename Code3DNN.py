import os
import pandas as pd
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#Now perpare the data:
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder


class SimpleDeepNeuralNetwork(nn.Module):
    def __init__(self):
        super(SimpleDeepNeuralNetwork, self).__init__()
        self.net = nn.Sequential(
        nn.Linear(10, 32),
        ##nn.ReLU(),
        ##nn.Linear(32, 32),
        ##nn.ReLU(),
        ##nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 4)
        )        

    def forward(self, x):        
        return self.net(x)

# Create an instance of the network
def trainDNN( _X_train, _X_test, _Y_train, _Y_test):
    le = LabelEncoder()

    X_train= torch.from_numpy(_X_train.values, ).to(torch.float32)
    Y_train= torch.from_numpy(le.fit_transform(_Y_train))
    X_test= torch.from_numpy(_X_test.values).to(torch.float32)
    Y_test= torch.from_numpy(le.fit_transform(_Y_test))
    

    model=SimpleDeepNeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    #Train
    for epoch in range(2000):    
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    #Evaluate
    with torch.no_grad():
        preds = torch.argmax(model(X_test), dim=1)
        accuracy = (preds == Y_test).float().mean()
        print(f"Test Accuracy: {accuracy:.2f}")
    return accuracy


model=SimpleDeepNeuralNetwork()
print(model)