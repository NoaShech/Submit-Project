import os
import cv2
import numpy as np
import json
from ultralytics import YOLO
from keras.models import load_model
import matplotlib.pyplot as plt

def letterbox_image(img, desired_size=256):
    h, w = img.shape[:2]
    ratio = float(desired_size) / max(h, w)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    delta_w = desired_size - new_w
    delta_h = desired_size - new_h
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left
    
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

def predict_frame(image_rgb):
    yolo_model_path = "yolov8n_taco.pt"
    trash_classifier_path = r"C:\Users\User\Desktop\Noa Project\trash_classifier_taco_cropped.h5"

    yolo_model = YOLO(yolo_model_path)
    trash_model = load_model(trash_classifier_path)
    trash_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    image_with_boxes = image_rgb.copy()
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    results = yolo_model.predict(source=image_bgr, conf=0.25, verbose=False)
    r = results[0]
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        return []
    
    bboxes = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    conf_threshold = 0.5
    filtered_boxes = [(bbox.astype(int), float(conf)) for bbox, conf in zip(bboxes, confidences) if conf >= conf_threshold]

    trash_classes = {0: "cardboard", 1: "glass", 2: "metal", 3: "paper", 4: "plastic", 5: "trash"}

    detections_list = []
    for idx, (bbox, conf_det) in enumerate(filtered_boxes):
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - 5)
        y1 = max(0, y1 - 5)
        x2 = min(image_rgb.shape[1], x2 + 5)
        y2 = min(image_rgb.shape[0], y2 + 5)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_resized = letterbox_image(crop, desired_size=256)
        crop_input = crop_resized.astype("float32") / 255.0
        crop_input = np.expand_dims(crop_input, axis=0)

        prediction = trash_model.predict(crop_input)
        pred_class_idx = int(np.argmax(prediction, axis=1)[0])
        confidence_cls = float(prediction[0][pred_class_idx])
        predicted_label = trash_classes.get(pred_class_idx, "unknown")

        detections_list.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "yolo_confidence": conf_det,
            "class_confidence": confidence_cls,
            "class_label": predicted_label
        })

    return detections_list

if __name__ == '__main__':
    # עדכני את הנתיב לתמונת הדוגמה שלך
    sample_image_path = r"C:\Users\User\Desktop\Test\sample.jpg"
    if not os.path.exists(sample_image_path):
        print("Sample image not found at:", sample_image_path)
        exit(1)
    image = cv2.imread(sample_image_path)
    if image is None:
        print("Error reading sample image")
        exit(1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = predict_frame(image_rgb)
    
    # ציור תיבות גבול ותוויות על התמונה
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class_label']} {det['class_confidence']:.2f}"
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image_rgb, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title("Detection Results")
    plt.show()
