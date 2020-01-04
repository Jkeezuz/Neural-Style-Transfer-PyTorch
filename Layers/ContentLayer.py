
import torch.nn as nn
import torch.nn.functional as F


class ContentLayer(nn.Module):
    """
    Define the custom content layer used to compute the content loss
    In init it gets as input the activation values for content image
    In forward method it gets as input the activations for generated image
    """
    def __init__(self, target_activations, ):
        super(ContentLayer, self).__init__()
        # Detach the target content from the computation graph to ensure
        # it stays constant and does not throw errors during computation
        self.target_activations = target_activations.detach()
        self.loss = 0

    def forward(self, generated_activations):
        # Compute the loss as Mean Squared Errors as in the original paper Gatys et. al. (2015)
        self.loss = F.mse_loss(generated_activations, self.target_activations)
        # Return the activations to ensure it is passed forward in the network
        return generated_activations
