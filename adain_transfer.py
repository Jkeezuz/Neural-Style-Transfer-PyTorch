from torch.utils.data import DataLoader

from dataprocess.StyleTransferDataset import StyleTransferDataset
from utilities import *
from AdaIN import AdaIN
import pprint


# DEBUG ONLY
def rename(directory):
    import os

    for i, filename in enumerate(sorted(os.listdir(directory))):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            ext = os.path.splitext(filename)[1]
            os.rename(os.path.join(directory, filename), os.path.join(directory, str(i)+ext))


# FOR TEST PURPOSES
if __name__ == "__main__":
    # DEBUG ONLY
    # rename(CONTENT_PATH)
    
    style_layers_req = ["Conv2d_1", "Conv2d_2", "Conv2d_3", "Conv2d_4"]

    adain = AdaIN(4, style_layers_req)

    pprint.pprint(adain.encoder)
    pprint.pprint(adain.decoder)

    # TRAIN
    transformed_dataset = StyleTransferDataset(CONTENT_PATH, STYLE_PATH, transform=transforms.Compose([
                                               transforms.Resize(256),
                                               transforms.RandomCrop(224),
                                               transforms.ToTensor()]))
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)

    adain.train(dataloader=dataloader, style_weight=1000, epochs=500)
