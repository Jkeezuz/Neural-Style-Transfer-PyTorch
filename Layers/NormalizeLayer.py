# Create the normalization layer
# As stated in the pytorch neural style transfer tutorial:
# "VGG networks are trained on images with each channel normalized by
# mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]"

import torch.nn as nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NormalizeLayer(nn.Module):
    """
    Layer used to normalize the input data.
    Useful for transfer learning to ensure that our data is
    normalized in the same way as the training data for model was
    """
    def __init__(self, mean, std):
        super(NormalizeLayer, self).__init__()
        # Reshape the mean and std to [C, 1, 1] to work with pytorch tensors of shape [N, C, H, W]
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, image):
        # Normalize the image
        return (image - self.mean) / self.std

