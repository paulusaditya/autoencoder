import os
import numpy as np
import matplotlib.pyplot as plt
from src.model import build_autoencoder
from tensorflow.keras.callbacks import ModelCheckpoint

def train_model(x_train, y_train, x_val, y_val, output_dir="outputs/saved_model", epochs=50):
    model = build_autoencoder(x_train.shape[1:])
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = ModelCheckpoint(f"{output_dir}/best_model.h5", save_best_only=True)
    
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        shuffle=True,
                        validation_data=(x_val, y_val),
                        callbacks=[checkpoint])
    
    # Save loss plot
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.savefig(f"{output_dir}/loss.png")
    return model
