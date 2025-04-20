"""
This script trains a trash classifier using Transfer Learning on cropped TACO images.
We use MobileNetV2 as the base model (pretrained on ImageNet) and fine-tune the last 30 layers.
The input images are resized to 256x256.
The model classifies images into 6 trash categories.
After training, the model is saved as 'trash_classifier_taco_cropped.h5'.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_finetuned_model(input_shape=(256, 256, 3), num_classes=6):
    # Load MobileNetV2 with ImageNet weights, without the top layers
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    
    # Freeze all layers initially
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze the last 30 layers for fine-tuning
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)  # Prevent overfitting
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile with a lower learning rate for fine-tuning
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def main():
    # Path to the cropped TACO dataset (structure: TacoCropped/plastic, metal, paper, glass, cardboard, trash)
    dataset_dir = r"C:\Users\User\Desktop\Noa Project\Taco\TACO-master\data\TacoCropped"
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode="categorical",
        subset="training"
    )
    
    val_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode="categorical",
        subset="validation"
    )

    print("val_generator.class_indices:", val_generator.class_indices)

    # הפיכת המילון class_indices למילון הפוך: אינדקס -> שם קטגוריה
    idx_to_class = {v: k for k, v in val_generator.class_indices.items()}
    print("Index to class mapping from val generator:", idx_to_class)

    model = create_finetuned_model(input_shape=(256,256,3), num_classes=6)
    model.summary()
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        verbose=1
    )
    
    model_save_path = "trash_classifier_taco_cropped.h5"
    model.save(model_save_path)
    print(f"Model trained on cropped TACO dataset and saved as '{model_save_path}'.")
    
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
