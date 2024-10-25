import os

import dill

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# Check Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Here I reworked the original code to fit the pretrained AlexNet network input requirements.
# See: https://pytorch.org/hub/pytorch_vision_alexnet/

# Transformation steps for input data 
AlexTransform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), # images need to be RGB, MNIST is in greyscale. Therefore needs to be converted to RGB
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(), # Also brings the tensor values in the range [0, 1] instead of [0, 255]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Path to mnist data
mnist_data_path = os.path.join("..", "data", 'mnist-data')

# Create a loader for training data and testing data
train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(mnist_data_path, train=True, download=True, transform=AlexTransform),
        batch_size=500, shuffle=True)

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(mnist_data_path, train=False, download=True, transform=AlexTransform),
        batch_size=1, shuffle=False)

val_loader = torch.utils.data.DataLoader(
        datasets.MNIST(mnist_data_path, train=False, download=True, transform=AlexTransform),
        batch_size=500, shuffle=False)


# Load pretrained model
mod = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

# Change last layer to reflect MNIST categories
mod_fc = mod.classifier[-1] 
num_ftrs = mod_fc.in_features # Get the amount of input features for the last layer

# Update to 10 possible outputs
mod.classifier[-1] = nn.Linear(num_ftrs, 10)

# Freeze all layers first
for param in mod.parameters():
    param.requires_grad=False

# Unfreeze all fully connected layers
for layer in mod.classifier:
    if isinstance(layer, torch.nn.Linear):  # Check if the layer is a fully connected layer
        for param in layer.parameters():
            param.requires_grad = True  # Unfreeze the fully connected layer

# Create an instance of the model and send to GPU if available
net = mod.to(device)
# Define softmax function
log_softmax = nn.LogSoftmax(dim=1)

def model(x_data, y_data):

    fc1Layer_w = Normal(loc=torch.ones_like(net.classifier[1].weight), scale=torch.ones_like(net.classifier[1].weight))
    fc1Layer_b = Normal(loc=torch.ones_like(net.classifier[1].bias), scale=torch.ones_like(net.classifier[1].bias))

    fc2Layer_w = Normal(loc=torch.ones_like(net.classifier[4].weight), scale=torch.ones_like(net.classifier[4].weight))
    fc2Layer_b = Normal(loc=torch.ones_like(net.classifier[4].bias), scale=torch.ones_like(net.classifier[4].bias))

    fc3Layer_w = Normal(loc=torch.ones_like(net.classifier[6].weight), scale=torch.ones_like(net.classifier[6].weight))
    fc3Layer_b = Normal(loc=torch.ones_like(net.classifier[6].bias), scale=torch.ones_like(net.classifier[6].bias))

    priors = {'classifier[1].weight': fc1Layer_w,
              'classifier[1].bias': fc1Layer_b,
              'classifier[4].weight': fc2Layer_w,
              'classifier[4].bias': fc2Layer_b,
              'classifier[6].weight': fc3Layer_w,
              'classifier[6].bias': fc3Layer_b}

    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", net, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()

    lhat = log_softmax(lifted_reg_model(x_data))

    pyro.sample("obs", Categorical(logits=lhat), obs=y_data)
    
    
softplus = torch.nn.Softplus()

def guide(x_data, y_data):

    # First fully connected layer weight distribution priors
    fc1w_mu = torch.randn_like(net.classifier[1].weight)
    fc1w_sigma = torch.randn_like(net.classifier[1].weight)
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1Layer_w = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param).independent(1)

    # First fully connected layer bias distribution priors
    fc1b_mu = torch.randn_like(net.classifier[1].bias)
    fc1b_sigma = torch.randn_like(net.classifier[1].bias)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1Layer_b = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)

    # Second fully connected layer weight distribution priors
    fc2w_mu = torch.randn_like(net.classifier[4].weight)
    fc2w_sigma = torch.randn_like(net.classifier[4].weight)
    fc2w_mu_param = pyro.param("fc2w_mu", fc2w_mu)
    fc2w_sigma_param = softplus(pyro.param("fc2w_sigma", fc2w_sigma))
    fc2Layer_w = Normal(loc=fc2w_mu_param, scale=fc2w_sigma_param).independent(1)

    # Second fully connected layer bias distribution priors
    fc2b_mu = torch.randn_like(net.classifier[4].bias)
    fc2b_sigma = torch.randn_like(net.classifier[4].bias)
    fc2b_mu_param = pyro.param("fc2b_mu", fc2b_mu)
    fc2b_sigma_param = softplus(pyro.param("fc2b_sigma", fc2b_sigma))
    fc2Layer_b = Normal(loc=fc2b_mu_param, scale=fc2b_sigma_param)

    # Third fully connected layer weight distribution priors
    fc3w_mu = torch.randn_like(net.classifier[6].weight)
    fc3w_sigma = torch.randn_like(net.classifier[6].weight)
    fc3w_mu_param = pyro.param("fc3w_mu", fc3w_mu)
    fc3w_sigma_param = softplus(pyro.param("fc3w_sigma", fc3w_sigma))
    fc3Layer_w = Normal(loc=fc3w_mu_param, scale=fc3w_sigma_param).independent(1)

    # Third fully connected layer bias distribution priors
    fc3b_mu = torch.randn_like(net.classifier[6].bias)
    fc3b_sigma = torch.randn_like(net.classifier[6].bias)
    fc3b_mu_param = pyro.param("fc3b_mu", fc3b_mu)
    fc3b_sigma_param = softplus(pyro.param("fc3b_sigma", fc3b_sigma))
    fc3Layer_b = Normal(loc=fc3b_mu_param, scale=fc3b_sigma_param)

    priors = {'classifier[1].weight': fc1Layer_w,
              'classifier[1].bias': fc1Layer_b,
              'classifier[4].weight': fc2Layer_w,
              'classifier[4].bias': fc2Layer_b,
              'classifier[6].weight': fc3Layer_w,
              'classifier[6].bias': fc3Layer_b}

    lifted_module = pyro.random_module("module", net, priors)

    return lifted_module()

optim = Adam({'lr': 0.01})
svi = SVI(model, guide, optim, loss=Trace_ELBO())

def evaluate_model(val_loader):
    correct = 0
    total = 0
    loss = 0
    net.eval()  # Set the model to evaluation mode
    
    
    with torch.no_grad():  # Disable gradient calculation
      for batch_id, data in enumerate(val_loader):
        x_val, y_val = data[0].to(device), data[1].to(device)
        
        # Get predictions from sampled models
        yhats = [guide(None, None)(x_val) for _ in range(5)]
        avg_yhat = torch.mean(torch.stack(yhats), dim=0)  # Average over the sampled models
        
        # Get predicted classes
        _, predicted = torch.max(avg_yhat, 1)
        total += y_val.size(0)  # Total samples
        correct += (predicted == y_val).sum().item()  # Correct predictions
        
        # Accumulate loss for each batch
        loss += svi.evaluate_loss(x_val, y_val)
    
    net.train()  # Switch back to training mode after evaluation
    
    # Normalize loss value
    normalizer_val = len(val_loader.dataset)
    total_epoch_loss_val = loss / normalizer_val
    
    # Calculate accuracy
    total_epoch_accuracy = round(correct / total * 100, 2)
    
    return (total_epoch_loss_val, total_epoch_accuracy)

# Loading the model and parameters without using the regular loading function
# This is because parameters had to be stored using dill package, instead of pickle.
# Saving the parameters in the normal way resulted in errors due to weakref objects (which cannot be serialized by pickle)

load_model = True

model_num = "03"

if load_model:
    # Load model
    load_model_path = os.path.join("results", "pretrained_models", "model_" + model_num + ".pt")
    saved_model_dict = torch.load(load_model_path) 
    net.load_state_dict(saved_model_dict['model'])
    guide = saved_model_dict['guide']
    
    # Load parameters
    load_path_params = os.path.join("results", "pretrained_models", "model_" + model_num + "_params.pt")
    with open(load_path_params, 'rb') as file:
        loaded_params = dill.load(file)
    
    # Store parameters from loaded file into parameter store.
    for param_name, param_tensor in loaded_params.items():
        pyro.param(param_name, param_tensor)
        

net.eval()
sampled_models = [guide(None, None) for _ in range(5)]

sample_data, labels = next(iter(test_loader))
print("True label:", labels)
# Predicted class
yhats = sampled_models[1](sample_data.to(device))

# Graph predictions
x_vals = np.array([i for i in range(10)])
plt.bar(x_vals, yhats.squeeze().cpu().detach().numpy())
plt.xlabel("Digit")
plt.ylabel("Log Probability?")
plt.show()


