import os
import numpy as np
from PIL import Image

def load_images_from_folder(folder_path, size=(48, 48)):
    images = []
    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                try:
                    img = Image.open(img_path).convert("L").resize(size)
                    img = np.array(img) / 255.0
                    images.append(img)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    images = np.array(images)
    images = np.expand_dims(images, axis=-1)  # (H, W, 1)
    return images
