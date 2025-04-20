"""
קובץ זה מאמן מודל סיווג מסוג Transfer Learning עבור זיהוי סוגי פסולת.
אנו משתמשים במודל MobileNetV2 המאומן מראש על ImageNet כבסיס,
ומוסיפים עליו שכבות מותאמות לסיווג ל-6 הקטגוריות:
plastic, metal, paper, glass, cardboard, trash.
לאחר האימון, המודל נשמר לקובץ 'trashnet_classifier_finetuned.h5'.
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

def create_finetuned_model():
    # טוענים את MobileNetV2 עם משקולות ImageNet וללא השכבות העליונות
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    
    # קובע שהשכבות של הבסיס יהיו קפואות (לא מתעדכנות בתחילת האימון)
    for layer in base_model.layers:
        layer.trainable = False

    # הוספת שכבות מותאמות
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)  # Dropout למניעת overfitting
    predictions = Dense(6, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    # קומפילציה של המודל עם אופטימייזר Adam
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def main():
    # נתיב לדאטה סט TrashNet - ודא שהדאטה סט מסודר לפי תיקיות (תת-תיקיות לכל קטגוריה)
    trashnet_dataset = r"C:\Users\User\Desktop\Noa Project\TrashNet\dataset-resized"
    
    # יצירת ImageDataGenerator עם אוגמנטציות
    data_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,  # 80% אימון, 20% ולידציה
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    
    # יצירת generator לאימון
    train_gen = data_gen.flow_from_directory(
        trashnet_dataset,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    # יצירת generator לוולידציה
    val_gen = data_gen.flow_from_directory(
        trashnet_dataset,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    # יצירת המודל המאומן באמצעות Transfer Learning
    model = create_finetuned_model()
    
    # אימון המודל
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50
    )
    
    # שמירת המודל המאומן
    model.save("trashnet_classifier_finetuned.h5")
    print("Model fine-tuned on TrashNet and saved as 'trashnet_classifier_finetuned.h5'.")
    
    # הצגת גרפים של Loss ו-Accuracy
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
