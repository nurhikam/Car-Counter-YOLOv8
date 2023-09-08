from ultralytics import YOLO
import cv2
import cvzone
import math

# For Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

# Using Videos
# cap = cv2.VideoCapture('../Videos/cars.mp4')

model = YOLO("../Yolo-Weights/yolov8l.pt")

className = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
             "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
             "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
             "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
             "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
             "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
             "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
             "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
             "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
             "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
             "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
             "toothbrush"
             ]

while True:
    success, img = cap.read()
    result = model(img, stream=True)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 3)

            # cvZone
            w, h = x2-x1, y2-y1
            bbox = x1, y1, w, h
            cvzone.cornerRect(img, bbox)
            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            # Class Name
            cls = int(box.cls[0])

            #Display Confidence & Class
            cvzone.putTextRect(img, f'{className[cls]} {conf}', (max(0, x1+15), max(35, y1-15)), scale= 2, thickness= 2)



    cv2.imshow("Image", img)
    cv2.waitKey(1)

    #pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
