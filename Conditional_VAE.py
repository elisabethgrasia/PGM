# -*- coding: utf-8 -*-
"""
Created on Mon Jun 5 08:24:52 2023

@author: elisabethgp
"""

import torch, os
from torch import nn, optim
from torch.nn import functional as F

import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils 
import torch.utils.data 
import matplotlib.pyplot as plt

# Parameters
input_size = 784 # 28x28
hidden_size = 144
num_epochs = 20
batch_size = 100
learning_rate = 0.001
rho = 0.05 # Sparsity parameter
beta = 3 # Weight of sparsity penalty term

# Autoencoder model
class CVAutoencoder(nn.Module):
    def __init__(self, img_dim, hid_dim, latent_dim, nclasses):
        super(CVAutoencoder, self).__init__()
        self.nclasses = nclasses
        self.latent_dim = latent_dim

        # Encoder
        self.fc1 = nn.Linear(self.input_dim, hid_dim)
        self.fc_mu = nn.Linear(hid_dim, latent_dim)
        self.fc_logvar = nn.Linear(hid_dim, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim + nclasses, hid_dim)
        self.fc4 = nn.Linear(hid_dim, img_dim)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))
    
    def concat_label(self, x, labels):
        labels = F.one_hot(labels, num_classes = self.nclasses)
        x = torch.cat((x, labels), dim= 1)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar
    
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nclasses = len(train_dataset.classes)
    
model = CVAutoencoder(input_size, hidden_size).to(device)    

# Loss and optimiser
criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Train the model
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs, mu, logvar = model(images, labels)
        mse_loss = criterion(outputs, images, mu, logvar)
        
        
        # Total loss
        loss = mse_loss 
        
        # Backward and optimise
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, loss.item()))
            