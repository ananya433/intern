from ultralytics import YOLO
import cv2, os
import requests
from datetime import datetime
model,cap= YOLO('YOLOv8n.pt'), cv2.VideoCapture(0)
os.makedirs('snapshots', exist_ok=True)
last_save=0

def get_location_by_ip():
    try:
        response = requests.get('https://ipapi.co/json')
        if response.status_code == 200:
            data = response.json()
            return{
                'city': data.get('city','unknown'),
                'region': data.get('region','unknown'),
                'country': data.get('country_name','unknown'),
                'latitude': data.get('latitude',0),
                'longitude': data.get('longitude',0)
            }
            
    except:
        pass
    return{'city':'unknown','region':'unknown','country':'unknown','latitude':0,'longitude':0}

while True:
    ret, frame = cap.read()
    if not ret: break
    r = model(frame,classes=[0,67])[0]
    person_count = sum (int(b.cls[0]) == 0 for b in r.boxes)
    cellphone_count = sum (int(b.cls[0]) == 67 for b in r.boxes)
    frame = r.plot()
    cv2.putText(frame, f'Persons: {person_count}  Cellphones: {cellphone_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    current_time = datetime.now().timestamp()

    # if (current_time - last_firebase_update) >= 5:
    #     send_to_firebase(person_count)
    #     last_firebase_update = current_time
    if person_count > 2 and person_count< 7 and  (datetime.now().timestamp() - last_save) >= 3:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cv2.rectangle(frame, (0, 0), (128, 60), (0,255,255), -1)
        # cv2.circle(frame,center=(64, 30), radius=30,color=(255,255,0),thickness=3) 
        cv2.putText(frame, f"{timestamp} | persons:{person_count}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"{get_location_by_ip()['city']}, {get_location_by_ip()['country']}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imwrite(f"snapshots/{datetime.now().strftime('%Y%m%d_%H%M%S')},.jpg", frame)
        cv2.imwrite(f"snapshots/{get_location_by_ip()['city']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", frame)
        print(f" saved snapshot")
        last_save = datetime.now().timestamp()
        
    cv2.imshow("PersonDetection", frame)   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
