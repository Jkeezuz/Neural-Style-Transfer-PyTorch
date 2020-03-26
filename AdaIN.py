from time import strftime, gmtime

import torch.nn as nn
import torchvision.models as models

import copy
import torch.optim as optim

from Layers.NormalizeLayer import NormalizeLayer
from Layers.AdainStyleLayer import AdainStyleLayer

from utilities import *

import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]


class AdaIN(object):

    def __init__(self, depth, style_req):
        self.style_layers = []
        self.encoder = self.build_encoder(depth, style_req)
        self.decoder = self.build_decoder(depth)

    def build_encoder(self, depth, style_req):
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

            # Check for style layers
            if name in style_req:
                # Create the style layer
                style_layer = AdainStyleLayer()
                # Append it to the module
                model.add_module("StyleLayer_{}".format(i), style_layer)
                self.style_layers.append(style_layer)

        return model.eval()

    def build_decoder(self, depth):
        """Decoder mirrors the encoder architecture"""
        # TODO: FOR NOW WE ASSUME DEPTH = 4

        model = nn.Sequential().train()

        # Build decoder for depth = 4
        model.add_module("ConvTranspose2d_1", nn.ConvTranspose2d(128, 128, (3, 3), (1, 1), (1, 1)))
        model.add_module("ReLU_1", nn.ReLU(True))

        model.add_module("ConvTranspose2d_2", nn.ConvTranspose2d(128, 64, (3, 3), (1, 1), (1, 1)))
        model.add_module("Upsample_2", nn.Upsample(scale_factor=2))
        model.add_module("ReLU_2", nn.ReLU(True))

        model.add_module("ConvTranspose2d_3", nn.ConvTranspose2d(64, 64, (3, 3), (1, 1), (1, 1)))
        model.add_module("ReLU_3", nn.ReLU(True))

        model.add_module("ConvTranspose2d_4", nn.ConvTranspose2d(64, 3, (3, 3), (1, 1), (1, 1)))
        model.add_module("ReLU_4", nn.ReLU(True))

        # Send model to CUDA or CPU
        return model.to(device)

    def adain(self, style_features, content_features):
        """Based on section 5. of https://arxiv.org/pdf/1703.06868.pdf"""
        # Pytorch shape - NxCxHxW
        # Computing values across spatial dimensions
        # Compute std of content_features
        content_std = torch.std(content_features, [2, 3], keepdim=True)
        # Compute mean of content_features
        content_mean = torch.mean(content_features, [2, 3], keepdim=True)
        # Compute std of style_features
        style_std = torch.std(style_features, [2, 3], keepdim=True)
        # Compute mean of style_features
        style_mean = torch.mean(style_features, [2, 3], keepdim=True)

        return style_std * ((content_features - content_mean)/content_std) + style_mean

    def forward(self, style_image, content_image):
        # Encode style and content image
        style_encoding = self.encoder(style_image.to(device))
        content_encoding = self.encoder(content_image.to(device))

        # Compute AdaIN
        adain_result = self.adain(style_encoding, content_encoding)

        # Decode to image
        image_result = self.decoder(adain_result)

        # return image and adain result
        return image_result, adain_result

    def compute_style_loss(self, style, decoded):
        # Compute std and mean of input
        input_std = torch.std(style, [2, 3], keepdim=True)
        input_mean = torch.mean(style, [2, 3], keepdim=True)
        # Compute std and mean of target
        target_std = torch.std(decoded, [2, 3], keepdim=True)
        target_mean = torch.mean(decoded, [2, 3], keepdim=True)
        return F.mse_loss(input_mean, target_mean) + \
            F.mse_loss(input_std, target_std)

    def compute_loss(self, decoded_image, style_image, adain_result):
        # Pass decoded image through encoder
        gen_encoding = self.encoder(decoded_image)

        # Content loss, L2 norm
        content_loss = torch.dist(adain_result, gen_encoding)

        # Style Loss
        style_activations = []
        decoded_activations = []

        # Get the style image activations from network
        self.encoder(style_image.to(device))
        for sl in self.style_layers:
            style_activations.append(sl.activations)

        # Get the decoded image activations from network
        self.encoder(decoded_image.to(device))
        for sl in self.style_layers:
            decoded_activations.append(sl.activations)

        # Compute the cumulative value of style loss
        style_loss = 0
        for da, sa in zip(decoded_activations, style_activations):
            style_loss += self.compute_style_loss(da, sa)

        return style_loss, content_loss

    def train(self, dataloader, style_weight, epochs):

        opt = optim.Adam(self.decoder.parameters())

        style_losses = []
        content_losses = []

        for epoch in range(epochs):
            for i_batch, sample in enumerate(dataloader):

                opt.zero_grad()

                decoded, adain_res = self.forward(sample['style'], sample['content'])
                style_loss, content_loss = self.compute_loss(decoded, sample['style'], adain_res)
                total_loss = style_loss*style_weight + content_loss

                total_loss.backward()

                opt.step()

                # Check network performance every x steps
                if i_batch == 0:
                    test, _ = self.forward(sample['style'], sample['content'])
                    show_tensor(test, epoch)
                    print("Epoch {0} at {1}:".format(epoch, strftime("%Y-%m-%d %H:%M:%S", gmtime())))
                    print('Style Loss(w/ style weight) : {:4f} Content Loss: {:4f}'.format(
                        style_loss.item()*style_weight, content_loss.item()))
                    print()

                    # Plot the loss
                    style_losses.append(style_loss.item()*style_weight)
                    content_losses.append(content_loss.item())
                    plt.figure()
                    plt.plot(range(epoch+1), style_losses, label="style loss")
                    plt.plot(range(epoch+1), content_losses, label="content loss")
                    plt.legend()
                    plt.savefig('loss.png')
                    plt.close()

        # Save decoder after training
        torch.save(self.decoder.state_dict(), "decoder.pth")

