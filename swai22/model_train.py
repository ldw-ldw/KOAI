import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_csv('frost_dataset.csv')

input_data = df.iloc[:, :-1].values
labels = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(input_data, labels, test_size=0.2, shuffle=True)

X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

class frostClassifier(nn.Module):
    def __init__(self, input_size):
        super(frostClassifier, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

input_size = X_train.shape[1]
model = frostClassifier(input_size)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    train_predictions = model(X_train_tensor)
    test_predictions = model(X_test_tensor)
    
    train_loss = criterion(train_predictions, y_train_tensor)
    test_loss = criterion(test_predictions, y_test_tensor)
    
    train_accuracy = ((train_predictions >= 0.5).float() == y_train_tensor).float().mean()
    test_accuracy = ((test_predictions >= 0.5).float() == y_test_tensor).float().mean()
    
    print(f'\nTrain Loss: {train_loss.item():.4f}')
    print(f'Test Loss: {test_loss.item():.4f}')
    print(f'Train Accuracy: {train_accuracy.item():.4f}')
    print(f'Test Accuracy: {test_accuracy.item():.4f}')

torch.save(model.state_dict(), 'frost_classifier_model.pth')