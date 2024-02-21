from ultralytics import YOLO
import cv2

model = YOLO('C:/data/Documents/Car Counter YOLOv8/Yolo-Weights/yolov8l.pt')
results = model("C:/data/Documents/Car Counter YOLOv8/Running YOLO/Images/4.png", show=True)
cv2.waitKey(0)

