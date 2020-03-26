import os

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

size = 300


def resize(src_path, dest_path):

    for i, filename in enumerate(sorted(os.listdir(src_path))):
        # Convert image to RGB during opening
        image = Image.open(os.path.join(src_path, filename)).convert("RGB")

        # Resize so we can load more to memory
        if image.size[0] > size:
            image = image.resize((size, image.size[1]))
        if image.size[1] > size:
            image = image.resize((image.size[0], size))

        # Save to dest path
        image.save(os.path.join(dest_path, filename))


if __name__ == "__main__":

    resize("../data/train/content", "../data/train/content_processed")
