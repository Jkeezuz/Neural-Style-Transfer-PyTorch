import os

from PIL import Image


def resize(src_path, dest_path):

    for i, filename in enumerate(sorted(os.listdir(src_path))):
        # Convert image to RGB during opening
        image = Image.open(filename).convert("RGB")

        # Resize so we can load more to memory
        image.resize((800, 800))

        # Save to dest path
        image.save(os.path.join(dest_path, filename))
