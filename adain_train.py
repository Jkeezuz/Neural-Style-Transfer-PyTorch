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
    # TRAIN
    transformed_dataset = StyleTransferDataset(CONTENT_PATH, STYLE_PATH, transform=transforms.Compose([
                                               transforms.Resize(256),
                                               transforms.RandomCrop(224),
                                               transforms.ToTensor()]))
    dataloader = DataLoader(transformed_dataset, batch_size=8,
                            shuffle=True, num_workers=4)

    adain = AdaIN()

    pprint.pprint(adain.encoder)
    pprint.pprint(adain.decoder)

    # Save the random weights for reuse
    torch.save(adain.decoder.state_dict(), "decoder_random.pth")

    # Train with different style weights to see the difference
    style_weights = [10e2, 10e3, 10e4]
    for sw in style_weights:
        adain.train(dataloader=dataloader, style_weight=sw, epochs=5)
        # Reset decoder to starting weights
        adain.decoder.load_state_dict(torch.load("decoder_random.pth"))
