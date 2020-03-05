import torchvision.models as models
import copy
import pprint
# TODO: DELETE THIS LATER!!
from torch.utils.data import DataLoader

from dataprocess.StyleTransferDataset import StyleTransferDataset
from style_transfer import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from Layers.NormalizeLayer import NormalizeLayer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]


class AdaIN(object):

    def __init__(self, depth, style_req, style_image):
        self.style_layers = []
        self.encoder = self.build_encoder(depth, style_req, style_image)
        self.decoder = self.build_decoder(depth)

    def build_encoder(self, depth, style_req, style_image):
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
                # Get the style activations in this layer
                style_activations = model(style_image).detach()
                # Create the style layer
                style_layer = StyleLayer(style_activations)
                # Append it to the module
                model.add_module("StyleLayer_{}".format(i), style_layer)
                self.style_layers.append(style_layer)

        return model.eval()

    def build_decoder(self, depth):
        """Decoder mirrors the encoder architecture"""
        # TODO: FOR NOW WE ASSUME DEPTH = 4

        model = nn.Sequential().train().to(device)

        # Build decoder for depth = 4
        #model.add_module("ReLU_1", nn.ReLU())
        model.add_module("ConvTranspose2d_1", nn.ConvTranspose2d(128, 128, (3, 3), (1, 1), (1, 1)))

        model.add_module("ReLU_2", nn.ReLU())
        model.add_module("ConvTranspose2d_2", nn.ConvTranspose2d(128, 64, (3, 3), (1, 1), (1, 1)))
        model.add_module("Upsample_2", nn.Upsample(scale_factor=2))

        model.add_module("ReLU_3", nn.ReLU())
        model.add_module("ConvTranspose2d_3", nn.ConvTranspose2d(64, 64, (3, 3), (1, 1), (1, 1)))

        model.add_module("ReLU_4", nn.ReLU())
        model.add_module("ConvTranspose2d_4", nn.ConvTranspose2d(64, 3, (3, 3), (1, 1), (1, 1)))

        # Send model to CUDA or CPU
        return model

    def adain(self, style_features, content_features):
        """Based on section 5. of https://arxiv.org/pdf/1703.06868.pdf"""
        # Compute std of content_features
        content_std = torch.std(content_features, [1, 2], keepdim=True)
        # Compute mean of content_features
        content_mean = torch.mean(content_features, [1, 2], keepdim=True)
        # Compute std of style_features
        style_std = torch.std(style_features, [1, 2], keepdim=True)
        # Compute mean of style_features
        style_mean = torch.mean(style_features, [1, 2], keepdim=True)

        return style_std * ((content_features - content_mean)/content_std) + style_mean

    def forward(self, style_image, content_image):
        # Encode style and content image
        style_features = self.encoder(style_image)
        content_features = self.encoder(content_image)

        # Compute AdaIN
        adain_result = self.adain(style_features, content_features)

        # Decode to image
        image_result = self.decoder(adain_result)

        # return image and adain result
        return image_result, adain_result

    def compute_loss(self, decoded_image, style_img, adain_result):
        gen_encoding = self.encoder(decoded_image)

        # Content loss, L2 norm
        content_loss = torch.dist(adain_result, gen_encoding)

        # Style loss
        style_loss = 0
        self.encoder(style_img)
        for sl in self.style_layers:
            style_loss += sl.loss

        return style_loss, content_loss

    def train(self, dataloader, style_weight, epochs):

        opt = optim.Adam(self.decoder)

        for epoch in range(epochs):

            for style, content in enumerate(dataloader):
                opt.zero_grad()

                decoded, adain_res = self.forward(style, content)
                style_loss, content_loss = self.compute_loss(decoded, style, adain_res)
                total_loss = style_weight*style_loss + content_loss

                total_loss.backward()

                opt.step()

                # Check network performance every x steps
                if epoch % 10 == 0:
                    test, _ = self.forward(style, content)
                    show_tensor(test, epoch)

        # Save model after training
        # Save encoder
        torch.save(self.encoder.state_dict(), "encoder.pth")
        # Save decoder
        torch.save(self.decoder.state_dict(), "decoder.pth")


## FOR TEST PURPOSES
if __name__ == "__main__":
    style_layers_req = ["Conv2d_1", "Conv2d_2", "Conv2d_3", "Conv2d_4", "Conv2d_5", "Conv2d_6", "Conv2d_7", "Conv2d_8"]
    style_name = "vcm"
    style_tensor = image_loader(IMAGES_PATH+"{}.jpg".format(style_name))

    adain = AdaIN(4, style_layers_req, style_tensor)

    pprint.pprint(adain.encoder)
    pprint.pprint(adain.decoder)

    # Check forward method

    # Load the images as preprocessed tensors
    content_name = "sydopera1"
    style_name = "vcm"
    content_tensor = image_loader(IMAGES_PATH + "{}.jpg".format(content_name))
    style_tensor = image_loader(IMAGES_PATH + "{}.jpg".format(style_name))

    # Assert that they're same size
    assert content_tensor.size() == style_tensor.size()

    show_tensor(content_tensor, "Content")

    show_tensor(style_tensor, "Style")

    input_tensor = content_tensor.clone()

    output, _ = adain.forward(style_tensor, content_tensor)

    show_tensor(output, title="output")

    # TRAIN
    content_dir = ""
    style_dir = ""
    transformed_dataset = StyleTransferDataset(content_dir, style_dir, transform=transforms.Compose([
                                               transforms.Resize(256),
                                               transforms.RandomCrop(224),
                                               transforms.ToTensor()]))
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)

    adain.train(dataloader=dataloader, style_weight=10000, epochs=500)
