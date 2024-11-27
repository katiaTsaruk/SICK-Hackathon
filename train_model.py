"""
 * @author Ricardo Merlos Torres
 * @email [contact@ricardomerlostorres.com]
 * @create date 2024-11-26 00:18:24
 * @modify date 2024-11-26 00:18:24
 * @desc [description]
"""


from ultralytics import YOLO

model = YOLO('yolov8n.pt')  

# Train the model
model.train(
    data='./raspberry_dataset.yaml', 
    epochs=20,                    
    imgsz=640,                    
    batch=16,                     
    device="cpu"                    
)
