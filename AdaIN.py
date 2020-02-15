import torchvision.models as models
import copy
import pprint


import torch
import torch.nn as nn

from Layers.NormalizeLayer import NormalizeLayer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]


class AdaIN(object):

    def __init__(self, depth):
        self.encoder = self.build_encoder(depth)
        self.decoder = self.build_decoder(depth)

    def build_encoder(self, depth):
        """Builds an encoder that uses first "depth" numbers of convolutional layers of VGG19"""
        vgg = models.vgg19(pretrained=True).features.to(device)

        model_copy = copy.deepcopy(vgg)
        # Define the layer names from which we want to pick activations

        # Create a new model that will be modified version of given model
        # starts with normalization layer to ensure all images that are
        # inserted are normalized like the ones original model was trained on
        norm_layer = NormalizeLayer(normalize_mean, normalize_std).to(device)

        model = nn.Sequential(norm_layer).eval()
        model = model.to(device)

        i = 0
        # Loop over the layers
        for layer in model_copy.children():
            # The layers in vgg are not numerated so we have to add numeration
            # to copied layers so we can append our content and style layers to it
            name = ""
            # Check which instance this layer is to name it appropiately
            if isinstance(layer, nn.Conv2d):
                i += 1
                # Stop when we reach required depth
                if i > depth:
                    break
                name = "Conv2d_{}".format(i)
            if isinstance(layer, nn.ReLU):
                name = "ReLu_{}".format(i)
                layer = nn.ReLU(inplace=False)
            if isinstance(layer, nn.MaxPool2d):
                if i >= depth:
                    break
                name = "MaxPool2d_{}".format(i)
            # Layer has now numerated name so we can find it easily
            # Add it to our model
            model.add_module(name, layer)

        return model

    def build_decoder(self, depth):
        """Decoder mirrors the encoder architecture"""
        # TODO: FOR NOW WE ASSUME DEPTH = 4

        model = nn.Sequential().train()

        # Send model to CUDA or CPU
        model = model.to(device)

        # Build decoder for depth = 4
        model.add_module("ReLU_1", nn.ReLU())
        model.add_module("ConvTranspose2d_1", nn.ConvTranspose2d(128, 128, (3, 3), (1, 1), (1, 1)))

        model.add_module("ReLU_2", nn.ReLU())
        model.add_module("ConvTranspose2d_2", nn.ConvTranspose2d(128, 64, (3, 3), (1, 1), (1, 1)))
        model.add_module("MaxUnpool2d_2", nn.MaxUnpool2d(kernel_size=2, stride=2))

        model.add_module("ReLU_3", nn.ReLU())
        model.add_module("ConvTranspose2d_3", nn.ConvTranspose2d(64, 64, (3, 3), (1, 1), (1, 1)))

        model.add_module("ReLU_4", nn.ReLU())
        model.add_module("ConvTranspose2d_4", nn.ConvTranspose2d(64, 3, (3, 3), (1, 1), (1, 1)))

        return model

    def adain(self, content_features, style_features):
        """Based on section 5. of https://arxiv.org/pdf/1703.06868.pdf"""
        # Compute std of content_features
        content_std = content_features.std
        # Compute mean of content_features
        content_mean = content_features.mean
        # Compute std of style_features
        style_std = style_features.std
        # Compute mean of style_features
        style_mean = style_features.mean

        return style_std * ((content_features - content_mean)/content_std) + style_mean


# test
if __name__ == "__main__":

    adain = AdaIN(4)

    pprint.pprint(adain.encoder)
    pprint.pprint(adain.decoder)

