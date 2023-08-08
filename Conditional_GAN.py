# -*- coding: utf-8 -*-
"""
Created on Mon Jun 5 08:13:25 2023

@author: elisabethgp
"""

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import os


os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Generator(nn.Module):
    def __init__(self, latent_dim, img_dim, nclasses):
        super(Generator, self).__init__()
        self.input_dim = latent_dim + nclasses
        self.nclasses = nclasses
        self.gen = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )
        
    def forward(self, x, label):
        label = F.one_hot(label, self.n_classes)
        x = torch.cat((x, label), dim=1)
        return self.gen(x)
    
class Discriminator(nn.Module):
    def __init__(self, img_dim, nclasses):
        super(Discriminator, self).__init__()
        self.input_dim = latent_dim + nclasses
        self.nclasses = nclasses
        self.disc = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x, label):
        label = F.one_hot(label, self.nclasses)
        x = torch.cat((x, label), dim=1)
        return self.disc(x)

# Hyperparameters

latent_dim = 64
img_x = img_y = 28
img_dim = img_x * img_y # Flattened MNIST 28x28 = 784
batch_size = 32
lr = 0.001
epochs = 50

# Initialize generator and discriminator
gen = Generator(latent_dim, img_dim, n_classes, 256).to(device)        
disc = Discriminator(img_dim, n_classes, 256).to(device)

# optimisers
opt_gen = torch.optim.Adam(gen.parameters(), lr=lr)
opt_disc = torch.optim.Adam(disc.parameters(), lr=lr)

# Loss function
criterion = nn.BCELoss()


if isinstance(epochs, tuple) and len(epochs) > 1:
    start_epoch = epochs[0] - 1 # Previous iteration
    end_epoch = epochs[1]
elif isinstance(epochs, int):
    start_epoch = 0
    end_epoch = epochs
    epochs = start_epoch, end_epoch
     
# Load data
train_dataset = MNIST(root='.', transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
    ]), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

n_classes = len(train_dataset.classes)

# Initialize generator and discriminator
gen = Generator(latent_dim, img_dim, n_classes, 256).to(device)        
disc = Discriminator(img_dim, n_classes, 256).to(device)


# Training loop
for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(train_loader):
        real = real.view(-1, img_dim).to(device)
        batch_size = real.shape[0]
        
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, latent_dim).to(device)
        fake = gen(noise)
        
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward()
        
        # Train Generator : min log(1 - (D(G(z)))) <-> max log(D(G(z)))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()
        
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(train_loader)} \
                    Loss D: {lossD:.4f}, loss G: {lossG:.4f}")
                    
            with torch.no_grad():
                fake = gen(noise).reshape(-1, 1, img_x, img_y)
                data = real.reshape(-1, 1, img_x, img_y)
                img_grid_fake = make_grid(fake, normalize=True)
                img_grid_real = make_grid(data, normalize = True)
                
                plt.figure(figsize=(10, 10))
                plt.subplot(1, 2, 1)
                plt.title("Real Images")
                plt.imshow(img_grid_real.permute(1, 2, 0).cpu().numpy())
                plt.subplot(1, 2, 2)
                plt.title("Fake Images")
                plt.imshow(img_grid_fake.permute(1, 2, 0).cpu().numpy())
                plt.show()