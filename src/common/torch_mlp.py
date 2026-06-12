import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random

class TorchMLP(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, num_classes=3, activation='tanh'):
        super(TorchMLP, self).__init__()
        layers = []
        in_features = input_size
        for h in hidden_layer_sizes:
            layers.append(nn.Linear(in_features, h))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            in_features = h
        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)
        
        # We don't initialize weights here, we will initialize them in fit using numpy's RandomState

    def forward(self, x):
        return self.network(x)

class TorchMLPWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(100,), activation='tanh', solver='adam', 
                 alpha=0.0001, learning_rate='constant', max_iter=15000, 
                 random_state=None, batch_size='auto', tol=1e-4, n_iter_no_change=10,
                 device='auto'):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.batch_size = batch_size
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.device = device
        
    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            random.seed(self.random_state)
            
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        num_classes = len(self.classes_)
        # Assuming classes are contiguous from 0. If not, needs mapping, but here they are 0,1,2.
        
        self.model_ = TorchMLP(n_features, self.hidden_layer_sizes, num_classes, activation=self.activation)
        
        if self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        else:
            batch_size = self.batch_size
            
        # Separate weights and biases to match sklearn's L2 penalty which only applies to weights
        weights = [p for n, p in self.model_.named_parameters() if 'weight' in n]
        biases = [p for n, p in self.model_.named_parameters() if 'bias' in n]
        
        optimizer = optim.Adam([
            {'params': weights, 'weight_decay': self.alpha / n_samples},
            {'params': biases, 'weight_decay': 0.0}
        ], lr=0.01)
        
        criterion = nn.CrossEntropyLoss()
        
        if self.device == 'auto' or self.device is None:
            self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device_ = torch.device(self.device)
            
        self.model_.to(self.device_)
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device_)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device_)
            
        best_loss = float('inf')
        no_improvement_count = 0
        
        indices = np.arange(n_samples)
        
        self.model_.train()
        for epoch in range(self.max_iter):
            epoch_loss = 0.0
            np.random.shuffle(indices)
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_idx = indices[start_idx:end_idx]
                
                batch_X = X_tensor[batch_idx]
                batch_y = y_tensor[batch_idx]
                
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            
            epoch_loss /= n_samples
            
            if best_loss - epoch_loss > self.tol:
                best_loss = epoch_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            if no_improvement_count >= self.n_iter_no_change:
                break
                
        return self

    def predict(self, X):
        self.model_.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device_)
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()
