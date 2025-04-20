import os
import json
import cv2
import shutil
from collections import defaultdict
import matplotlib.pyplot as plt

# נתיבים – עדכן בהתאם לסביבת העבודה שלך
TACO_DATA_PATH = r"C:\Users\User\Desktop\Noa Project\Taco\TACO-master\data"
ANNOTATIONS_FILE = os.path.join(TACO_DATA_PATH, "annotations.json")
# תיקיית הפלט לדאטה סט החדש של האובייקטים החתוכים
OUTPUT_DATASET_DIR = os.path.join(TACO_DATA_PATH, "TacoCropped")

# 6 הקטגוריות הרצויות (כפי ש־TrashNet משתמש)
TRASHNET_CATEGORIES = ["plastic", "metal", "paper", "glass", "cardboard", "trash"]

# יצירת תיקיות עבור כל קטגוריה במידת הצורך
for cat in TRASHNET_CATEGORIES:
    os.makedirs(os.path.join(OUTPUT_DATASET_DIR, cat), exist_ok=True)

# יצירת מילון לספירת התמונות לפי קטגוריה
category_counts = defaultdict(int)

# פונקציה למיפוי – קודם לפי שם, אחר כך לפי supercategory
def map_category_to_trashnet(name: str, supercat: str):
    name_lower = name.strip().lower()
    super_lower = supercat.strip().lower()
    
    # מילון מיפוי לפי שם (NAME_TO_CATEGORY)
    NAME_TO_CATEGORY = {
        "aluminium blister pack": "plastic",
        "carded blister pack": "plastic",
        "other plastic bottle": "plastic",
        "clear plastic bottle": "plastic",
        "plastic bottle cap": "plastic",
        "disposable plastic cup": "plastic",
        "foam cup": "plastic",
        "other plastic cup": "plastic",
        "plastic lid": "plastic",
        "plastic film": "plastic",
        "garbage bag": "plastic",
        "other plastic wrapper": "plastic",
        "single-use carrier bag": "plastic",
        "plastic container": "plastic",
        "foam food container": "plastic",
        "other plastic container": "plastic",
        "plastic utensils": "plastic",
        "plastic straw": "plastic",
        "styrofoam piece": "plastic",
        "crisp packet": "plastic",
        "six pack rings": "plastic",
        "spread tub": "plastic",
        "tupperware": "plastic",
        
        "metal bottle cap": "metal",
        "scrap metal": "metal",
        "pop tab": "metal",
        "drink can": "metal",
        "food can": "metal",
        "aerosol": "metal",
        
        "paper cup": "paper",
        "normal paper": "paper",
        "magazine paper": "paper",
        "tissues": "paper",
        "wrapping paper": "paper",
        "paper bag": "paper",
        "paper straw": "paper",
        
        "other carton": "cardboard",
        "egg carton": "cardboard",
        "drink carton": "cardboard",
        "corrugated carton": "cardboard",
        "meal carton": "cardboard",
        "pizza box": "cardboard",
        "toilet tube": "cardboard",
        
        "glass bottle": "glass",
        "broken glass": "glass",
        "glass jar": "glass",
        "glass cup": "glass",
        
        "cigarette": "trash",
        "food waste": "trash",
        "battery": "trash",
        "shoe": "trash",
    }
    if name_lower in NAME_TO_CATEGORY:
        return NAME_TO_CATEGORY[name_lower]
    
    # מילון מיפוי לפי supercategory (SUPER_TO_CATEGORY)
    SUPER_TO_CATEGORY = {
        "bottle": "plastic",
        "bottle cap": "plastic",
        "cup": "plastic",
        "carton": "cardboard",
        "can": "metal",
        "paper": "paper",
        "paper bag": "paper",
        "plastic bag & wrapper": "plastic",
        "plastic container": "plastic",
        "plastic glooves": "plastic",
        "plastic utensils": "plastic",
        "styrofoam piece": "plastic",
        "blister pack": "plastic",
        "other plastic": "trash",
        "cigarette": "trash",
        "food waste": "trash",
        "battery": "trash",
        "shoe": "trash",
        "rope & strings": "trash",
        "squeezable tube": "trash",
        "unlabeled litter": "trash",
    }
    for key, value in SUPER_TO_CATEGORY.items():
        if key in super_lower:
            return value
    return "trash"  # ברירת מחדל

# פונקציה לביצוע letterboxing – שינוי גודל תוך שמירה על יחס הממדים והוספת רקע שחור
def resize_keep_aspect(img, desired_size=224):
    old_size = img.shape[:2]  # (height, width)
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    resized = cv2.resize(img, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)
    
    # חשב את המרווחים להוספה (letterboxing)
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    # הוסף גבולות שחורים
    new_img = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return new_img

# טוען את קובץ האנוטציות
ANNOTATIONS_FILE = os.path.join(TACO_DATA_PATH, "annotations.json")
with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

images_info = data.get("images", [])
annotations_info = data.get("annotations", [])
categories_info = data.get("categories", [])

# בונה מילון: cat_id -> (name, supercategory)
cat_id_to_details = {}
for cat in categories_info:
    cat_id = cat["id"]
    cat_name = cat["name"]
    cat_super = cat["supercategory"]
    cat_id_to_details[cat_id] = (cat_name, cat_super)

# בונה מילון: image_id -> רשימת (cat_id, bbox)
image_id_to_objects = defaultdict(list)
for ann in annotations_info:
    image_id = ann["image_id"]
    cat_id = ann["category_id"]
    bbox = ann["bbox"]  # בפורמט [x, y, width, height]
    image_id_to_objects[image_id].append((cat_id, bbox))

print("Processing TACO images and extracting cropped objects...")

count_total = 0
count_saved = 0

# עבור כל תמונה בדאטה סט
for img_info in images_info:
    image_id = img_info["id"]
    file_name = img_info["file_name"]
    img_path = os.path.join(TACO_DATA_PATH, file_name)
    if not os.path.exists(img_path):
        continue
    # טוען את התמונה (באמצעות OpenCV)
    img = cv2.imread(img_path)
    if img is None:
        continue

    # עבור כל אובייקט (annotation) בתמונה זו
    annotations = image_id_to_objects.get(image_id, [])
    for i, (cat_id, bbox) in enumerate(annotations):
        count_total += 1
        x, y, w, h = bbox  # [x, y, width, height]
        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        # במקום למתוח (resize) את התמונה החתוכה, נעשה letterboxing כדי לשמר את יחס הממדים
        crop_processed = resize_keep_aspect(crop, desired_size=128)
        
        # קבלת פרטי הקטגוריה מהאנוטציה
        cat_name, cat_super = cat_id_to_details.get(cat_id, ("", ""))
        trashnet_cat = map_category_to_trashnet(cat_name, cat_super)
        
        # עדכון ספירת התמונות לכל קטגוריה
        category_counts[trashnet_cat] += 1
        
        # שמירה: יוצרים תיקייה בהתאם לקטגוריה (plastic, metal, וכו')
        out_dir = os.path.join(OUTPUT_DATASET_DIR, trashnet_cat)
        os.makedirs(out_dir, exist_ok=True)
        out_filename = f"{image_id}_{i}.jpg"
        out_path = os.path.join(out_dir, out_filename)
        cv2.imwrite(out_path, crop_processed)
        count_saved += 1

print(f"Processed {len(images_info)} images, extracted {count_total} objects, saved {count_saved} cropped images.")

# הצגת גרף עבור כמות התמונות החתוכות לפי קטגוריה
categories = list(category_counts.keys())
counts = list(category_counts.values())
plt.figure(figsize=(8,6))
plt.bar(categories, counts, color='skyblue')
plt.xlabel("Category")
plt.ylabel("Number of Cropped Images")
plt.title("Cropped Images per Category")
plt.show()
