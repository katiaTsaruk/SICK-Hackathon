"""
 * @author Ricardo Merlos Torres
 * @email [contact@ricardomerlostorres.com]
 * @create date 2024-11-25 20:58:34
 * @modify date 2024-11-25 20:58:34
 * @desc [SICK Hackathon YoloImplementation]
"""

import cv2
from ultralytics import YOLO
import numpy as np

#model = YOLO('yolov8n.pt')
model = YOLO('runs/detect/train8/weights/best.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


def is_bbox_inside(person_box, phone_box):
    px1, py1, px2, py2 = person_box
    cx1, cy1, cx2, cy2 = phone_box

    return cx1 >= px1 and cy1 >= py1 and cx2 <= px2 and cy2 <= py2

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    
    results = model(frame)

    
    #print(results[0].boxes)
    annotated_frame = results[0].plot()
    #print(annotated_frame)
    print(results[0])

    people = []
    cellphone = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        class_id = int(box.cls[0])

        if conf >= 0.5:
            if class_id == 0:
                people.append((x1, y1, x2, y2))
            elif class_id == 67:
                cellphone.append((x1, y1, x2, y2))

    print("People:", people)
    print("Cellphone:", cellphone)

    for x1, y1, x2, y2 in people:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    for x1, y1, x2, y2 in cellphone:
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    

    
    for person_box in people:
        for phone_box in cellphone:
            if is_bbox_inside(person_box, phone_box):
                px1, py1, px2, py2 = person_box
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)  
                cv2.putText(frame, "Person in a call", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else: 
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 3)
                cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    cv2.imshow("SICK Hackathon DEMO", annotated_frame)
    #cv2.imshow("SICK Hackathon DEMO", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
