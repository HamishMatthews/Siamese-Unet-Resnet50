import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import sys

# for reloading modules
import importlib

# import local modules
from siamese_resnet50_unet import SiameseUNetWithResnet50Encoder
from data_loading import FireDataset
from train_model import train_model

# Reload the module
importlib.reload(sys.modules[train_model.__module__])

# Now re-import the function
from train_model import train_model

# Provide your hdf5_file and fold values
hdf5_file = 'datasets/train_eval.hdf5'
train_folds = [0, 1, 2, 3]  # Replace with your train fold indices
test_folds = [4]  # Replace with your test fold indices

train_dataset = FireDataset(hdf5_file, train_folds)
test_dataset = FireDataset(hdf5_file, test_folds)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create a model
model = SiameseUNetWithResnet50Encoder()

# Define the loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Train the model
trained_model = train_model(model, train_dataloader, criterion, optimizer, scheduler, num_epochs=10)

# Save the trained model
torch.save(trained_model.state_dict(), 'path/to/trained_model.pth')
