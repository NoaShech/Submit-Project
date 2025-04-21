זוהי מערכת מבוססת דפדפן לזיהוי ומיון פסולת בזמן אמת בעזרת ראייה ממוחשבת ולמידת מכונה. המשתמש מצלם תמונה של אשפה בעזרת מצלמת הטלפון, והמערכת חוזה את סוג הפסולת ומציגה את התוצאה.

טכנולוגיות שבהן השתמשתי
Python
Flask – ליצירת שרת צד שרת
YOLOv8 – לזיהוי אובייקטים בתמונה
רשת עצבית מסוג CNN – לסיווג סוגי פסולת
HTML / CSS / JS – לבניית הממשק
DroidCam – לחיבור מצלמת הטלפון לאתר

איך להריץ את המערכת
הורידו את הקבצים או שיבטו (clone) את הריפוזיטורי.
ודאו ש־Python 3.10 ומעלה מותקן.
צרו סביבת פיתוח וירטואלית והפעילו אותה:
python -m venv myenv
myenv\Scripts\activate

התקינו את כל התלויות:

pip install -r requirements.txt

הורידו את הקבצים הכבדים (ראו למטה) והניחו אותם בתיקיות המתאימות.

הריצו את קובץ app_submit.py:

python app_submit.py

היכנסו בדפדפן לכתובת http://127.0.0.1:5000










# Trash Classification Web App 🌍🗑️

## 🇬🇧 English Overview

This project is a web-based application for real-time **trash classification and detection** using computer vision and deep learning. 
It enables users to capture images of trash using their phone camera and instantly receive predictions about the type of waste, helping sort it into the correct recycling bin.

### Technologies Used

- **Python**
- **Flask** – for creating the backend web server
- **YOLOv8** – for real-time object detection
- **CNN model** – for trash type classification
- **HTML/CSS/JS** – for building the web interface
- **DroidCam** – for connecting a mobile phone camera to the app

### How to Run the Project

1. Clone the repository or download it as a ZIP and extract it.
2. Make sure you have Python 3.10+ installed.
3. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv myenv
   myenv\Scripts\activate  # Windows

Install dependencies:
pip install -r requirements.txt

Download the missing large files (YOLO weights, .h5 model, datasets) from the links in the section below.
Run the Flask app:
python app_submit.py

Open your browser at http://127.0.0.1:5000







מאגר זה לא כולל את קובצי הדאטה סט המלאים ואת קבצי המודלים המאומנים בשל מגבלות גודל של GitHub. כדי להריץ את הפרויקט במלואו, יש להוריד את הקבצים הבאים באופן ידני:

TrashNet.zip – דאטה סט לסיווג פסולת
TACO.zip – דאטה סט עם תיוגים לזיהוי אובייקטים
trash_classifier_taco_cropped.h5 – מודל סיווג (CNN) מאומן
yolov8n.pt, yolov8n_taco.pt – מודלי YOLOv8 לזיהוי אובייקטים



This repository does not include the full datasets and trained model files due to size limitations. To run the project end-to-end, you will need to download the following files manually:

TrashNet.zip – Dataset for trash classification (https://drive.google.com/file/d/16hYv82WGZg-vkeaCiR8z_574xvxjaL6v/view?usp=sharing)
TACO.zip – Annotated dataset for object detection (https://drive.google.com/file/d/1uiuchu3wh1U3eYxQWiyLfjDpZ8I-2sMG/view?usp=sharing)
trash_classifier_taco_cropped.h5 – Trained CNN classifier model
yolov8n.pt, yolov8n_taco.pt – YOLOv8 detection models


