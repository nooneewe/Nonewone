from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to YOLOv8 API!"}

model = YOLO("C:/Users/ASUS/Desktop/yolov8-fastapi/best.pt")

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # قراءة الصورة وتحويلها إلى مصفوفة OpenCV
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # تنفيذ التنبؤ
    results = model(img)

    # استخراج الأسماء فقط مرتبة من اليسار إلى اليمين
    detections = []
    for box in results[0].boxes:
        x1 = box.xyxy[0][0].item()
        cls_name = model.names[box.cls[0].item()]
        detections.append((x1, cls_name))
    
    # ترتيب حسب الإحداثي الأيسر
    detections_sorted = sorted(detections, key=lambda x: x[0])
    
    # إزالة التكرارات مع الحفاظ على الترتيب
    seen = set()
    unique_classes = []
    for _, cls_name in detections_sorted:
        if cls_name not in seen:
            seen.add(cls_name)
            unique_classes.append(cls_name)
    
    # دمج الأسماء في سلسلة واحدة
    result_str = "".join(unique_classes)

    return {"result": result_str}