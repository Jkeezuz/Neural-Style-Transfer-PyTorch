
import torch.nn as nn
import torch.nn.functional as F
import torch

class AdainStyleLayer(nn.Module):
    """
    Custom style layer used to access the feature space
    of previous neural network layer and compute the style loss
    for input image
    """
    def __init__(self):
        super(AdainStyleLayer, self).__init__()
        # Compute the gram matrix of target activations for style image
        self.target_activations = None
        self.loss = 0
        self.target = False

    def forward(self, activations):

        self.target_activations = activations
        # Pass the activations forward in neural network
        return activations
