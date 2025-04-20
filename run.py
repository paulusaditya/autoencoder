from src.data_loader import load_images_from_folder
from src.train import train_model
import os

# Load train and test images
x_train = load_images_from_folder("dataset/train", size=(48, 48))
x_test = load_images_from_folder("dataset/test", size=(48, 48))

# Karena ini autoencoder reconstruction
y_train = x_train
y_test = x_test

# Train
model = train_model(x_train, y_train, x_test, y_test)
