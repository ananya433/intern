from ultralytics import YOLO
import cv2
import easyocr
import numpy as np

# Initialize YOLO model and OCR reader
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'], gpu=False)

cap = cv2.VideoCapture(0)

def extract_plate_text(image):
    try:                                 
        # Pre-process for better OCR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray)
        
        # Get all detected text
        plates = []
        for (bbox, text, conf) in results:
            if conf > 0.3 and len(text) >= 4:
                # Clean text
                text = ''.join(c for c in text if c.isalnum() or c == ' ')
                plates.append((text.upper(), conf))
        
        return plates
    except:
        return []


frame_count = 0
plates = []

while True:
    ret, frame = cap.read() {}
    if not ret:
        break
    
    frame_count += 1
    
    frame = cv2.resize(frame, (640, 480))
    
    # Only run OCR every 10 frames (much faster)
    if frame_count % 10 == 0:
        plates = extract_plate_text(frame)
    
    # Display detected plates (from last OCR scan)
    y_pos = 40
    for text, conf in plates:
        accuracy = int(conf * 100)
        cv2.putText(frame, f"{text} ({accuracy}%)", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 35
        if frame_count % 10 == 0:  # Only print when OCR runs
            print(f"✓ Detected: {text} | Accuracy: {accuracy}%")
    
    cv2.putText(fram             aaaaa  
       
