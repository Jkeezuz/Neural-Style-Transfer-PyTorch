from torch.utils.data import DataLoader

from dataprocess.StyleTransferDataset import StyleTransferDataset
from resources.utilities import *
from Model.AdaIN import AdaIN
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
    
    style_layers_req = ["ReLu_1", "ReLu_2", "ReLu_3", "ReLu_4"]

    # TRAIN
    transformed_dataset = StyleTransferDataset(CONTENT_PATH, STYLE_PATH, transform=transforms.Compose([
                                               transforms.Resize(256),
                                               transforms.RandomCrop(224),
                                               transforms.ToTensor()]))
    dataloader = DataLoader(transformed_dataset, batch_size=8,
                            shuffle=True, num_workers=4)

    adain = AdaIN(4, style_layers_req)

    pprint.pprint(adain.encoder)
    pprint.pprint(adain.decoder)

    # Save the random weights for reuse
    torch.save(adain.decoder.state_dict(), "decoder_random.pth")

    for sw in [100, 1000, 10000]:
        adain.train(dataloader=dataloader, style_weight=sw, epochs=10)

        # Reset decoder to starting weights
        adain.decoder.load_state_dict(torch.load("decoder_random.pth"))
