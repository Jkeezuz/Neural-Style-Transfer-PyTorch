import os

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def resize(src_path, dest_path):

    for i, filename in enumerate(sorted(os.listdir(src_path))):
        # Convert image to RGB during opening
        image = Image.open(os.path.join(src_path, filename)).convert("RGB")

        # Resize so we can load more to memory
        if image.size[0] > 800:
            image = image.resize((800, image.size[1]))
        if image.size[1] > 800:
            image = image.resize((image.size[0], 800))

        # Save to dest path
        image.save(os.path.join(dest_path, filename))


if __name__ == "__main__":

    resize("../data/train/style", "../data/train/style_processed")
