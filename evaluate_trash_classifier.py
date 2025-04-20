import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

def evaluate_model():
    # Path to the cropped TACO dataset – folder structure as before
    dataset_dir = r"C:\Users\User\Desktop\Noa Project\Taco\TACO-master\data\TacoCropped"
    batch_size = 32
    img_size = (256, 256)  # updated size to match model input
    
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    val_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    print("val_generator.class_indices:", val_generator.class_indices)

    # הפיכת המילון class_indices למילון הפוך: אינדקס -> שם קטגוריה
    idx_to_class = {v: k for k, v in val_generator.class_indices.items()}
    print("Index to class mapping from val generator:", idx_to_class)

    model_path = r"C:\Users\User\Desktop\Noa Project\trash_classifier_taco_cropped.h5"
    model = load_model(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    predictions = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes
    class_labels = list(val_generator.class_indices.keys())
    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print("Classification Report:")
    print(report)
    
    # גרף מטריצת בלבול
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    
    # גרף דיוק לכל קטגוריה (accuracy per class)
    class_accuracy = []
    for i in range(len(class_labels)):
        total = np.sum(cm[i, :])
        correct = cm[i, i]
        acc = correct / total if total > 0 else 0
        class_accuracy.append(acc * 100)
    
    plt.figure(figsize=(8, 6))
    plt.bar(class_labels, class_accuracy, color='green')
    plt.xlabel("Category")
    plt.ylabel("Accuracy (%)")
    plt.title("Class-wise Accuracy")
    plt.ylim(0, 100)
    plt.show()

if __name__ == "__main__":
    evaluate_model()
