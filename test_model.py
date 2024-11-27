"""
 * @author Ricardo Merlos Torres
 * @email [contact@ricardomerlostorres.com]
 * @create date 2024-11-26 00:21:33
 * @modify date 2024-11-26 00:21:33
 * @desc [description]
"""


from ultralytics import YOLO

model = YOLO('yolov8n.pt')

metrics = model.val(data='./raspberry_dataset.yaml')