import torch.utils.data.Dataset as Dataset
import os, os.path
import torchvision.datasets as dataset
import torchivsion.transforms as transforms

class StyleTransferDataset(Dataset):
    """ Dataset designed for Style Transfer with AdaIN method """

    def __init__(self, content_root_dir, style_root_dir, transform=None):
        """

        :param content_root_dir: path to directory containing content images
        :param style_root_dir:  path to directory containing style images
        :param transform: optional transform to be apllied on a sample
        """
        self.content_root_dir = content_root_dir
        self.style_root_dir = style_root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.content_root_dir))

    def __getitem__(self, idx):

        content_name = os.path.join(self.content_root_dir, idx+".jpg")
        style_name = os.path.join(self.style_root_dir, idx+".jpg")