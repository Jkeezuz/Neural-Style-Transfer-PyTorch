
import torch.nn as nn
import torch.nn.functional as F
import torch


def gram_matrix(activations):
    """
    Compute the Gram matrix of filters to extract the style
    :param activations: features from neural network layer
    :return: Normalized Gram matrix
    """
    # Get the shape of activations
    n, c, h, w = activations.size()
    # n = batch size
    # c = number of channels
    # h = height of activation
    # w = width of activation
    # Resize the activations to 2D matrix of size (n*c, h*w)
    activations_resized = activations.view((n * c, h * w))
    # Compute gram matrix
    G = torch.mm(activations_resized, activations_resized.t())
    # Normalize the matrix
    return G.div(n * c * h * w)


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
        # If this is pass for target activations (we're passing style image
        if self.target:
            self.target_activations = gram_matrix(activations).detach()
            self.loss = 0
        else:
            if self.target_activations is not None:
                # We're passing another image, should update loss
                # Compute the gram matrix for generated activations
                G = gram_matrix(activations)
                # Compute the style loss
                self.loss = F.mse_loss(G, self.target_activations)

        # Pass the activations forward in neural network
        return activations
