from time import strftime, gmtime

import torch.nn as nn
import torchvision.models as models

import torch.optim as optim
from torch.backends import cudnn

from Layers.NormalizeLayer import NormalizeLayer

from resources.utilities import *

import torch.nn.functional as F

cudnn.benchmark = True
cudnn.enabled = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]


class AdaIN(object):

    def __init__(self):
        self.style_layers = []
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        """Builds an encoder that uses first 4 numbers of convolutional layers of VGG19"""

        class Encoder(nn.Module):
            """Vgg 19 up to ReLU4_1 modified to return intermediate results (activations)"""

            def __init__(self):
                super(Encoder, self).__init__()

                self.norm = NormalizeLayer(normalize_mean, normalize_std)

                vgg = list(models.vgg19(pretrained=True).to(device).eval().features)[:21]

                # Get desired features from vgg
                self.relu1 = nn.Sequential(*vgg[:2])  # ReLu1_1 output
                self.relu2 = nn.Sequential(*vgg[2:7])  # ReLU2_1 output
                self.relu3 = nn.Sequential(*vgg[7:12])  # ReLU3_1 output
                self.relu4 = nn.Sequential(*vgg[12:21])  # ReLU4_1 output

            def forward(self, x):
                # Normalization
                x = self.norm(x)

                # Forward pass
                x1 = self.relu1(x)
                x2 = self.relu2(x1)
                x3 = self.relu3(x2)
                x4 = self.relu4(x3)

                return [x1, x2, x3, x4]

        model = Encoder()
        return model

    def build_decoder(self):
        """Decoder mirrors the encoder architecture"""
        model = nn.Sequential()

        # Build decoder for depth = 4
        model.add_module("ConvTranspose2d1_1", nn.ConvTranspose2d(512, 256, (3, 3), (1, 1), (1, 1)))
        model.add_module("ReLU1_1", nn.ReLU())
        model.add_module("Upsample1_1", nn.Upsample(scale_factor=2))

        model.add_module("ConvTranspose2d2_1", nn.ConvTranspose2d(256, 256, (3, 3), (1, 1), (1, 1)))
        model.add_module("ReLU2_1", nn.ReLU())

        model.add_module("ConvTranspose2d2_2", nn.ConvTranspose2d(256, 256, (3, 3), (1, 1), (1, 1)))
        model.add_module("ReLU2_2", nn.ReLU())

        model.add_module("ConvTranspose2d2_3", nn.ConvTranspose2d(256, 256, (3, 3), (1, 1), (1, 1)))
        model.add_module("ReLU2_3", nn.ReLU())

        model.add_module("ConvTranspose2d2_4", nn.ConvTranspose2d(256, 128, (3, 3), (1, 1), (1, 1)))
        model.add_module("ReLU2_4", nn.ReLU())
        model.add_module("Upsample2_4", nn.Upsample(scale_factor=2))

        model.add_module("ConvTranspose2d3_1", nn.ConvTranspose2d(128, 128, (3, 3), (1, 1), (1, 1)))
        model.add_module("ReLU3_1", nn.ReLU())

        model.add_module("ConvTranspose2d3_2", nn.ConvTranspose2d(128, 64, (3, 3), (1, 1), (1, 1)))
        model.add_module("ReLU3_2", nn.ReLU())
        model.add_module("Upsample3_2", nn.Upsample(scale_factor=2))

        model.add_module("ConvTranspose2d4_1", nn.ConvTranspose2d(64, 64, (3, 3), (1, 1), (1, 1)))
        model.add_module("ReLU4_1", nn.ReLU())

        model.add_module("ConvTranspose2d4_2", nn.ConvTranspose2d(64, 3, (3, 3), (1, 1), (1, 1)))
        model.add_module("ReLU4_2", nn.ReLU())

        # Send model to CUDA or CPU
        return model.train().to(device)

    def adain(self, style_features, content_features):
        """Based on section 5. of https://arxiv.org/pdf/1703.06868.pdf"""
        # with torch.no_grad():
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

        return style_std * ((content_features - content_mean) / content_std) + style_mean

    def forward(self, style_image, content_image, alpha=1.0):
        with torch.no_grad():
            # Encode style and content image
            style_activations = self.encoder(style_image)

            content_encoding = self.encoder(content_image)[-1]
            # Compute AdaIN
            adain_result = self.adain(style_activations[-1], content_encoding)
            adain_result = alpha * adain_result + (1 - alpha) * content_encoding

        # Decode to image
        generated_image = self.decoder(adain_result)

        # return image and adain result
        return generated_image, adain_result, style_activations

    def compute_style_loss(self, style, generated):
        # Compute std and mean of input
        style_std = torch.std(style, [2, 3], keepdim=True)
        style_mean = torch.mean(style, [2, 3], keepdim=True)
        # Compute std and mean of target
        generated_std = torch.std(generated, [2, 3], keepdim=True)
        generated_mean = torch.mean(generated, [2, 3], keepdim=True)
        return F.mse_loss(style_mean, generated_mean) + \
               F.mse_loss(style_std, generated_std)

    def compute_loss(self, generated_image, adain_result, style_activations):
        # Get the decoded image activations from network
        gen_activations = self.encoder(generated_image)

        # Compute the cumulative value of style loss
        style_loss = 0
        for sa, da in zip(style_activations, gen_activations):
            style_loss += self.compute_style_loss(sa, da)

        # Content loss, L2 norm
        content_loss = torch.dist(adain_result, gen_activations[-1], 2)

        return style_loss, content_loss

    def train(self, dataloader, style_weight, epochs):

        def adjust_learning_rate(optimizer, lr, lr_decay, iteration_count):
            """Imitating the original implementation"""
            new_lr = lr / (1.0 + lr_decay * iteration_count)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        opt = optim.Adam(self.decoder.parameters(), lr=1e-4)

        style_losses = []
        content_losses = []
        for epoch in range(epochs):
            # adjust_learning_rate(opt, 1e-4, 5e-5, epoch)
            for i_batch, sample in enumerate(dataloader):

                content_image = sample['content'].to(device)
                style_image = sample['style'].to(device)

                opt.zero_grad()

                gen_image, adain, style_act = self.forward(style_image, content_image)
                style_loss, content_loss = self.compute_loss(gen_image, adain, style_act)
                style_loss = style_loss * style_weight
                total_loss = style_loss + content_loss

                total_loss.backward()

                opt.step()

                # Check network performance every x steps
                if i_batch == 0:
                    with torch.no_grad():
                        test, _, _ = self.forward(style_image, content_image)
                        show_tensor(content_image, "Content at epoch {0}".format(epoch), 1)
                        show_tensor(style_image, "Style at epoch {0}".format(epoch), 1)
                        show_tensor(test, "Style transfer at epoch {0}".format(epoch), 1)
                    print("Epoch {0} at {1}:".format(epoch, strftime("%Y-%m-%d %H:%M:%S", gmtime())))
                    print('Style Loss(w/ style weight) : {:4f} Content Loss: {:4f}'.format(
                        style_loss.item(), content_loss.item()))

                    # Plot the loss
                    style_losses.append(style_loss.item())
                    content_losses.append(content_loss.item())
                    plt.figure()
                    plt.plot(range(epoch + 1), style_losses, label="style loss")
                    plt.plot(range(epoch + 1), content_losses, label="content loss")
                    plt.legend()
                    plt.savefig('loss.png')
                    plt.close()

        # Save decoder after training
        torch.save(self.decoder.state_dict(), "decoder.pth")

    def load_save(self):
        self.decoder.load_state_dict(torch.load("decoder.pth"))
