import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import os, os.path
import torch

from PIL import Image


class StyleTransferDataset(Dataset):
    """ Dataset designed for Style Transfer with AdaIN method """

    def __init__(self, content_root_dir, style_root_dir, transform=None):
        """
        :param content_root_dir: path to directory containing content images
        :param style_root_dir:  path to directory containing style images
        :param transform: optional transform to be applied on a sample
        """
        self.content_root_dir = content_root_dir
        self.style_root_dir = style_root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.content_root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        content_name = os.path.join(self.content_root_dir, idx+".jpg")
        style_name = os.path.join(self.style_root_dir, idx+".jpg")

        content_image = Image.open(content_name)
        style_image = Image.open(style_name)

        sample = {'content': content_image, 'style': style_image}

        if self.transform:
            sample = self.transform(sample)

        return sample
