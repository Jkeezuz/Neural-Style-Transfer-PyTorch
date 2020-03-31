from torch.utils.data import DataLoader

from dataprocess.StyleTransferDataset import StyleTransferDataset
from resources.utilities import *
from Model.AdaIN import AdaIN
import pprint


# FOR TEST PURPOSES
if __name__ == "__main__":
    # DEBUG ONLY
    # rename(CONTENT_PATH)
    
    style_layers_req = ["Conv2d_1", "Conv2d_2", "Conv2d_3", "Conv2d_4"]

    # TRAIN
    transformed_dataset = StyleTransferDataset(CONTENT_PATH, STYLE_PATH, transform=transforms.Compose([
                                               transforms.Resize(256),
                                               transforms.RandomCrop(224),
                                               transforms.ToTensor()]))
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True)

    sample = next(iter(dataloader))

    adain = AdaIN(4, style_layers_req)
    adain.load_save()

    result, _ = adain.forward(sample['style'], sample['content'])
    show_tensor(sample['content'], "content")
    show_tensor(sample['style'], "style")
    show_tensor(result, "res")
