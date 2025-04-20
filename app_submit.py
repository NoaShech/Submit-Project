from flask import Flask, render_template, Response, request, send_file
import cv2
import numpy as np
import base64
import requests
import os
import time
from datetime import datetime

# ניסיון לייבא את מודול החיזוי, עם טיפול בשגיאות
try:
    from inference_pipeline_update import predict_frame
except ImportError:
    # פונקציית דמה במקרה שהמודול חסר
    def predict_frame(frame):
        print("Warning: Using dummy prediction function")
        # מחזיר רשימה ריקה או נתוני חיזוי לדוגמה למקרה שהמודול האמיתי חסר
        return [
            {
                "class_label": "פסולת כללית",
                "class_confidence": 0.95,
                "bbox": [100, 100, 200, 200]  # x1, y1, x2, y2
            }
        ]

# וודא שהנתיב לתבניות נכון
app = Flask(__name__, 
            template_folder='templates',  # נתיב לתיקיית התבניות
            static_folder='static')       # נתיב לתיקיית הקבצים הסטטיים
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

# הגדרת תיקיית snapshots
SNAPSHOT_FOLDER = 'static/snapshots'
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

# הגדרת מקורות מצלמה - קודם מקומית, אחר כך DroidCam
LOCAL_CAMERA_INDEX = 0  # מצלמה מקומית
DROIDCAM_URL = "http://192.168.1.116:4747/video"  # כתובת DroidCam

def get_camera():
    """
    מנסה לפתוח מצלמה בסדר העדיפויות:
    1. מצלמה מקומית
    2. DroidCam
    """
    # ניסיון 1: מצלמה מקומית
    print("Trying to connect to local camera...")
    cap = cv2.VideoCapture(LOCAL_CAMERA_INDEX)
    if cap.isOpened():
        print("Successfully connected to local camera")
        return cap
    
    # ניסיון 2: DroidCam
    print(f"Local camera failed. Trying DroidCam at {DROIDCAM_URL}...")
    cap = cv2.VideoCapture(DROIDCAM_URL)
    if cap.isOpened():
        print("Successfully connected to DroidCam")
        return cap
    
    print("Failed to connect to any camera")
    return None

def gen_frames():
    """יצירת מסגרות וידאו עבור הזרמת הווידאו"""
    cap = get_camera()
    if cap is None:
        # יצירת תמונת שגיאה במקום וידאו
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Camera Error", (200, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buf = cv2.imencode('.jpg', error_img)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buf.tobytes() + b'\r\n')
        return
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                # ניסיון לפתוח את המצלמה מחדש אם אבד חיבור
                cap.release()
                time.sleep(0.5)
                cap = get_camera()
                if cap is None:
                    break
                continue
                
            ret, buf = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buf.tobytes() + b'\r\n')
                   
            time.sleep(0.05)  # האט מעט את קצב הפריימים
    except Exception as e:
        print(f"Error in gen_frames: {e}")
    finally:
        # נקה את המשאבים
        if cap is not None:
            cap.release()

@app.route('/video_feed')
def video_feed():
    """הזנת וידאו חי למצלמה"""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    """צילום תמונה ושמירתה בקובץ"""
    # קבלת שם הקובץ מה-query string
    filename = request.args.get('filename', f'snapshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg')
    filepath = os.path.join(SNAPSHOT_FOLDER, filename)
    
    # פתיחת חיבור למצלמה
    cap = get_camera()
    if cap is None:
        return "Error: No camera available", 500
    
    # נסה מספר פעמים לצלם תמונה
    success = False
    for attempt in range(3):
        success, frame = cap.read()
        if success and frame is not None and frame.size > 0:
            break
        print(f"Capture attempt {attempt+1} failed, retrying...")
        time.sleep(0.5)
    
    # שחרור המצלמה
    cap.release()
    
    # בדוק שהצלחנו לקבל תמונה
    if not success or frame is None or frame.size == 0:
        return "Error capturing frame", 500
    
    # שמירת התמונה בתיקייה
    print(f"Saving snapshot to {filepath}")
    success = cv2.imwrite(filepath, frame)
    
    if not success:
        return f"Error saving image to {filepath}", 500
        
    print("Snapshot saved successfully")
    
    # החזרת התמונה כתגובה בלבד - החיזוי יתבצע בלחיצה על הכפתור
    ret, buf = cv2.imencode('.jpg', frame)
    return Response(buf.tobytes(), mimetype='image/jpeg')

@app.route('/predict')
def predict():
    """ביצוע חיזוי על תמונה מצולמת"""
    # קבלת שם הקובץ מה-query string
    filename = request.args.get('filename', '')
    
    if not filename:
        return "Error: No filename provided", 400
    
    filepath = os.path.join(SNAPSHOT_FOLDER, filename)
    
    if not os.path.exists(filepath):
        return f"Error: File not found: {filepath}", 404
    
    # קריאת התמונה מהקובץ
    print(f"Loading image from {filepath} for prediction")
    img = cv2.imread(filepath)
    if img is None or img.size == 0:
        return "Error: Cannot read image", 500
    
    # המרה ל-RGB לחיזוי
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # ביצוע החיזוי
    print("Performing prediction...")
    try:
        detections = predict_frame(rgb)
        print(f"Found {len(detections)} objects")
        
        # שרטוט התוצאות
        result_img = rgb.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class_label']} {det['class_confidence']:.2f}"
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(result_img, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # המרה חזרה ל-BGR ושמירה כתמונת תוצאה
        result_filename = f"result_{filename}"
        result_filepath = os.path.join(SNAPSHOT_FOLDER, result_filename)
        cv2.imwrite(result_filepath, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        
        # הכנת נתיב תמונה יחסי לתבנית
        result_path = f"/static/snapshots/{result_filename}"
        
        return render_template('result.html',
                            image_path=result_path,
                            detections=detections)
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"Error during prediction: {e}")
        print(error_msg)
        return f"Error during prediction: {str(e)}<br><pre>{error_msg}</pre>", 500

@app.route('/camera')
def camera():
    """עמוד המצלמה"""
    try:
        return render_template('camera.html')
    except Exception as e:
        print(f"Error rendering camera template: {e}")
        # נחזיר רשימה של התבניות הזמינות לצורך דיבאג
        import os
        templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
        available_templates = os.listdir(templates_dir) if os.path.exists(templates_dir) else []
        return f"Error: Camera template not found. Available templates: {available_templates}", 500

@app.route('/')
def index():
    """עמוד הבית"""
    return render_template('camera.html')

def ensure_templates_exist():
    """וודא שכל קבצי התבניות קיימים בתיקייה הנכונה"""
    import os
    
    # תוכן התבניות
    templates = {
        'camera.html': """<!-- templates/camera.html -->
<!DOCTYPE html>
<html lang="he">
<head>
  <meta charset="UTF-8">
  <title>מצלמה לזיהוי פסולת</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    /* מבנה גלובלי */
    html, body {
      margin: 0;
      padding: 0;
      height: 100vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      direction: rtl;
      font-family: Arial, sans-serif;
    }

    /* סרגל עליון */
    .header {
      background: #f8b8d0;
      padding: .5em;
      text-align: center;
    }
    .header h1 {
      margin: 0;
      font-size: 1.4em;
      color: #333;
    }

    /* כפתורי פעולה */
    .buttons {
      background: #fff;
      padding: .5em;
      text-align: center;
    }
    .buttons button {
      margin: .3em;
      padding: .6em 1.2em;
      font-size: 1em;
      cursor: pointer;
      border: none;
      background: #4CAF50;
      color: white;
      border-radius: 5px;
    }
    .buttons button:hover {
      background: #45a049;
    }

    /* מיכל הווידאו */
    .camera-container {
      flex: 1;
      position: relative;
      display: flex;
      justify-content: center;
      align-items: center;
      background: #000;
      width: 100%;
      height: 100%;
    }
    .camera-container img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }

    /* overlay לתצוגת ה‑snapshot */
    #snapshotContainer {
      position: absolute;
      top: 0; left: 0;
      width: 100%; height: 100%;
      display: none;
      justify-content: center;
      align-items: center;
      background: rgba(0,0,0,0.7);
      z-index: 10;
    }
    /* המסגרת הפנימית */
    #snapshotFrame {
      width: 80%;
      height: 70%;
      border: 4px solid #fff;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 2px 8px rgba(0,0,0,0.5);
      display: flex;
      justify-content: center;
      align-items: center;
      position: relative;
    }
    #snapshotFrame img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    
    /* כפתורי ה‑snapshot */
    #snapshotContainer .actions {
      position: absolute;
      bottom: 1em;
      display: flex;
      justify-content: center;
      width: 100%;
      gap: 1.5em;
    }
    #snapshotContainer .actions button {
      padding: .8em 1.5em;
      font-size: 1em;
      cursor: pointer;
      border: none;
      border-radius: 5px;
      font-weight: bold;
    }
    #predictBtn {
      background: #2196F3;
      color: white;
    }
    #predictBtn:hover {
      background: #1976D2;
    }
    #cancelBtn {
      background: #FF5722;
      color: white;
    }
    #cancelBtn:hover {
      background: #E64A19;
    }
    
    /* הסתרת הודעות שגיאה במקרה תקלה */
    #errorMessage {
      position: absolute;
      bottom: 1em;
      left: 0;
      right: 0;
      text-align: center;
      background: rgba(255, 0, 0, 0.8);
      color: white;
      padding: 0.5em;
      display: none;
    }
    
    /* חיווי טעינה */
    #loadingIndicator {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: rgba(0, 0, 0, 0.7);
      color: white;
      padding: 1em 2em;
      border-radius: 8px;
      display: none;
    }
  </style>
</head>
<body>

  <div class="header">
    <h1>זיהוי פסולת</h1>
  </div>

  <div class="buttons">
    <button id="captureBtn">📸 צלם תמונה</button>
    <button id="reloadBtn" onclick="location.reload()">🔄 רענן מצלמה</button>
  </div>

  <div class="camera-container" id="videoContainer">
    <!-- הווידאו החי -->
    <img id="videoFeed"
         src="{{ url_for('video_feed') }}"
         alt="Video Feed"
         onerror="console.error('Video feed error'); document.getElementById('errorMessage').style.display = 'block';"
    />

    <!-- overlay לתצוגת ה‑snapshot -->
    <div id="snapshotContainer">
      <div id="snapshotFrame">
        <img id="snapshot"
             src=""
             alt="Snapshot Preview"
             onload="console.log('Snapshot loaded:', this.src)"
             onerror="console.error('Snapshot load error:', this.src); document.getElementById('errorMessage').style.display = 'block';"
        >
        <div class="actions">
          <button id="predictBtn">🔍 זהה פסולת</button>
          <button id="cancelBtn">❌ ביטול</button>
        </div>
      </div>
    </div>
    
    <!-- הודעת שגיאה -->
    <div id="errorMessage">שגיאה בהתחברות למצלמה. נסה לרענן את הדף.</div>
    
    <!-- חיווי טעינה -->
    <div id="loadingIndicator">טוען...</div>
  </div>

  <script>
    const captureBtn       = document.getElementById('captureBtn');
    const videoFeedImg     = document.getElementById('videoFeed');
    const snapshotCont     = document.getElementById('snapshotContainer');
    const snapshotImg      = document.getElementById('snapshot');
    const predictBtn       = document.getElementById('predictBtn');
    const cancelBtn        = document.getElementById('cancelBtn');
    const errorMsg         = document.getElementById('errorMessage');
    const loadingIndicator = document.getElementById('loadingIndicator');
    let lastCaptureUrl     = "";
    let snapshotFilename   = "";

    captureBtn.addEventListener('click', () => {
      // יצירת שם קובץ ייחודי עבור ה-snapshot
      snapshotFilename = 'snapshot_' + Date.now() + '.jpg';
      
      // בונים URL מלא ל‑capture - בלי חיזוי אוטומטי, כי יש כפתורים
      lastCaptureUrl = window.location.origin + '/capture?filename=' + snapshotFilename;
      console.log('Taking snapshot:', lastCaptureUrl);

      // הסתרת הודעת שגיאה אם קיימת
      errorMsg.style.display = 'none';
      
      // הצגת חיווי טעינה
      loadingIndicator.textContent = 'מצלם תמונה...';
      loadingIndicator.style.display = 'block';
      
      // השבתת כפתור הצילום בזמן הפעולה
      captureBtn.disabled = true;

      // טוענים את ה‑snapshot
      fetch(lastCaptureUrl)
        .then(response => {
          if (!response.ok) {
            throw new Error('Network response was not ok');
          }
          return response.blob();
        })
        .then(blob => {
          const objectURL = URL.createObjectURL(blob);
          snapshotImg.src = objectURL;
          
          // מציגים את ה‑snapshot ומסתירים את הווידאו
          videoFeedImg.style.display = 'none';
          snapshotCont.style.display = 'flex';
          loadingIndicator.style.display = 'none';
          captureBtn.disabled = false;
        })
        .catch(error => {
          console.error('Error capturing snapshot:', error);
          errorMsg.textContent = 'שגיאה בצילום תמונה. נסה שוב מאוחר יותר.';
          errorMsg.style.display = 'block';
          loadingIndicator.style.display = 'none';
          captureBtn.disabled = false;
        });
    });

    // לחצן לביצוע חיזוי
    predictBtn.addEventListener('click', () => {
      // בדיקה שיש שם קובץ
      if (!snapshotFilename) {
        alert('לא נמצאה תמונה לחיזוי. אנא צלם תמונה תחילה.');
        return;
      }
      
      // הצגת חיווי טעינה
      loadingIndicator.textContent = 'מנתח תמונה...';
      loadingIndicator.style.display = 'block';
      
      // ניווט לעמוד חיזוי עם שם הקובץ
      window.location.href = window.location.origin + '/predict?filename=' + encodeURIComponent(snapshotFilename);
    });

    // לחצן ביטול וחזרה למצלמה
    cancelBtn.addEventListener('click', () => {
      // ביטול ויציאה לוידאו חי
      snapshotCont.style.display = 'none';
      videoFeedImg.style.display = 'block';
    });

    // רענון מיידי של הוידאו כשהדף נטען
    window.addEventListener('load', function() {
      // הוספת טיימסטמפ כדי למנוע קאשינג
      const timestamp = new Date().getTime();
      videoFeedImg.src = '{{ url_for('video_feed') }}?' + timestamp;
    });
  </script>

</body>
</html>""",
        'result.html': """<!DOCTYPE html>
<html lang="he">
<head>
  <meta charset="UTF-8">
  <title>תוצאות זיהוי פסולת</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {
      font-family: Arial, sans-serif;
      background-color: #fafafa;
      margin: 0;
      padding: 0;
      direction: rtl;
      text-align: center;
    }
    .header {
      background: #f8b8d0;
      padding: 0.8em;
      text-align: center;
      margin-bottom: 1em;
    }
    .header h1 {
      margin: 0;
      font-size: 1.4em;
      color: #333;
    }
    .result-container {
      margin: 1em auto;
      max-width: 800px;
      padding: 0 1em;
    }
    .result-image {
      max-width: 100%;
      border: 2px solid #ccc;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .detections {
      margin-top: 1.5em;
      text-align: right;
      background: #fff;
      padding: 1em;
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .detection-item {
      margin-bottom: 0.5em;
      padding-bottom: 0.5em;
      border-bottom: 1px solid #eee;
    }
    .detection-item:last-child {
      border-bottom: none;
    }
    .confidence {
      color: #008800;
      font-weight: bold;
    }
    .buttons {
      margin: 2em 0;
      display: flex;
      justify-content: center;
      gap: 1em;
      flex-wrap: wrap;
    }
    button {
      margin: 0.5em;
      padding: 0.8em 1.5em;
      font-size: 1.1em;
      cursor: pointer;
      border: none;
      color: white;
      border-radius: 5px;
      font-weight: bold;
      display: flex;
      align-items: center;
      gap: 0.5em;
      box-shadow: 0 2px 4px rgba(0,0,0,0.2);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    button:hover {
      transform: translateY(-2px);
      box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .camera-btn {
      background-color: #2196F3;
    }
    .camera-btn:hover {
      background-color: #1976D2;
    }
    .new-capture-btn {
      background-color: #4CAF50;
    }
    .new-capture-btn:hover {
      background-color: #45a049;
    }

    @media (max-width: 600px) {
      .buttons {
        flex-direction: column;
        align-items: center;
      }
      button {
        width: 80%;
      }
    }
  </style>
</head>
<body>
  <div class="header">
    <h1>תוצאות זיהוי פסולת</h1>
  </div>
  
  <div class="result-container">
    <img class="result-image" src="{{ image_path }}" alt="תמונה עם תוצאות זיהוי">
    
    <div class="detections">
      <h2>זוהו {{ detections|length }} פריטים</h2>
      
      {% if detections %}
        {% for det in detections %}
        <div class="detection-item">
          <strong>{{ det.class_label }}</strong> 
          <span class="confidence">({{ "%.2f"|format(det.class_confidence*100) }}%)</span>
          <div>מיקום: X1={{ det.bbox[0] }}, Y1={{ det.bbox[1] }}, X2={{ det.bbox[2] }}, Y2={{ det.bbox[3] }}</div>
        </div>
        {% endfor %}
      {% else %}
        <p>לא זוהו פריטים</p>
      {% endif %}
    </div>
    
    <div class="buttons">
      <button class="new-capture-btn" onclick="location.href='{{ url_for('camera') }}'">
        📸 צלם תמונה חדשה
      </button>
      <button class="camera-btn" onclick="window.history.back()">
        🔙 חזרה למצלמה
      </button>
    </div>
  </div>

  <script>
    // הדגשת כפתורי החזרה
    document.addEventListener('DOMContentLoaded', function() {
      const cameraButton = document.querySelector('.camera-btn');
      
      // אנימציה קלה להדגשת הכפתור
      setTimeout(() => {
        cameraButton.style.transform = 'scale(1.05)';
        setTimeout(() => {
          cameraButton.style.transform = 'scale(1)';
        }, 300);
      }, 1000);
    });
  </script>
</body>
</html>"""
    }
    
    # בדוק אם תיקיית התבניות קיימת, אם לא צור אותה
    template_dir = os.path.join(os.getcwd(), 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    # בדוק ועדכן כל תבנית
    for filename, content in templates.items():
        filepath = os.path.join(template_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created/Updated template: {filepath}")
    
    # וודא שתיקיית הסנאפשוטס קיימת
    snapshot_dir = os.path.join(os.getcwd(), 'static', 'snapshots')
    os.makedirs(snapshot_dir, exist_ok=True)
    print(f"Snapshot directory: {snapshot_dir}")

if __name__ == '__main__':
    # וודא שכל קבצי התבניות קיימים
    ensure_templates_exist()
    
    # הפעלת השרת בפורט 5000
    app.run(debug=True, port=5000)