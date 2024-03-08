import os
from PIL import Image


def resize_images(source_folder: str, destination_folder: str):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            img_path = os.path.join(source_folder, filename)
            img = Image.open(img_path)
            img_resized = img.resize((280, 280), Image.ANTIALIAS)
            img_resized.save(os.path.join(destination_folder, filename))

    print("All images have been resized and saved in the bws folder.")


if __name__ == "__main__":
    SOURCE_FOLDER = "./images/bw"
    DESTINATION_FOLDER = "./images/bws"
    resize_images(source_folder=SOURCE_FOLDER, destination_folder=DESTINATION_FOLDER)