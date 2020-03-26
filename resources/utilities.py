import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import  torchvision.transforms as transforms
import torch
from resources.constants import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -- UTILITY FUNCTIONS --


def resize(image, size):
    """Resize the image to size specified in the beginning of code"""
    return np.array(Image.fromarray(image).resize(size))


def image_loader(image_name):
    """Loads the images as preprocessed tensors"""
    # pytorch loader to take care of resizing and transforming image to tensor
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()])
    # Load the image from the disk
    image = Image.open(image_name)
    # Add batch dimension of 1 at dimension 0 to
    # satisfy pytorch's requirements of dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def to_image(tensor):
    """Converts tensor to PIL image"""
    # Converts to PIL image
    unloader = transforms.ToPILImage()
    # Clone the tensor to CPU
    image = tensor.cpu().clone()
    # Remove fake batch dimension
    image = image.squeeze(0)
    # Convert to PIL image
    image = unloader(image)
    return image


def show_tensor(tensor, title=None, num=None):
    """
    Helper function to convert the pytorch tensor back to displayable format.
    """
    # Turn on the interactive mode of plt
    plt.ion()
    plt.figure()

    sizes = list(tensor.size())
    if len(sizes) > 3:
        # Iterate over images
        for i in range(num if num else sizes[0]):
            image = to_image(tensor[i])

            plt.imshow(image)
            if title is not None:
                plt.title(title)

            plt.pause(0.001)

    else:
        image = to_image(tensor)

        plt.imshow(image)
        if title is not None:
            plt.title(title)

        plt.pause(0.001)

def save_tensor(tensor, title="NONAME"):
    """
    Helper function to save pytorch tensor as jpg image.
    """
    image = to_image(tensor)

    image.save(RESULTS_PATH+"{}.jpg".format(title))
