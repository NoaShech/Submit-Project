"""
This script trains a YOLOv8n object detection model on the TACO dataset.
The dataset is specified in the taco.yaml file.
After training, the model is saved for later use in the inference pipeline.
"""

from ultralytics import YOLO
import matplotlib.pyplot as plt

def main():
    # Path to the dataset configuration file (taco.yaml)
    data_config = r"C:\Users\User\Desktop\Noa Project\Taco\TACO-master\taco.yaml"
    
    # Initialize YOLOv8n with pretrained weights (from ultralytics)
    model = YOLO('yolov8n.pt')
    
    # Train the YOLO model on the TACO dataset.
    results = model.train(data=data_config, epochs=50, imgsz=640, save_period=5)
    
    # Save the trained model weights to a file.
    model.save("yolov8n_taco.pt")
    print("YOLOv8n model trained on TACO and saved as 'yolov8n_taco.pt'.")
    
    # הצגת גרף אימון (אם ultralytics מחזיר אובייקט עם היסטוריה)
    try:
        results.plot()   # ultralytics עשויה להכיל פונקציה זו להצגת גרפים
        plt.show()
    except Exception as ex:
        print("Could not plot training results:", ex)

if __name__ == "__main__":
    main()
