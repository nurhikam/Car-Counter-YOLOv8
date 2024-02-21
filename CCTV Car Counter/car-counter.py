from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# Import Videos
cap = cv2.VideoCapture('C:/data/Documents/Car Counter YOLOv8/Videos/Depan Mako Polres Purwakarta - Papais CCTV.mp4')

model = YOLO('C:/data/Documents/Car Counter YOLOv8/Yolo-Weights/yolov8l.pt')

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

mask = cv2.imread('C:/data/Documents/Car Counter YOLOv8/CCTV Car Counter/mask.png')

# Tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# position of line
limits = [830, 500, 1130, 500]

totalCount = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    detections = np.empty((0, 5))

    result = model(imgRegion, stream=True)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # cvZone
            w, h = x2-x1, y2-y1

            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            # Class Name
            cls = int(box.cls[0])

            # Display Confidence & Class just For Car
            currentClass = className[cls]
            if currentClass == 'car' or currentClass == 'truck' or currentClass == 'bus' and conf > 0.3 :
                # cvzone.putTextRect(img, f'{className[cls]} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=2, thickness=2, offset=10)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=2, colorR=(100, 100, 255), colorC=(255, 255, 0))
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (100, 100, 255), 2)

    for result in resultTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        # cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
        #                    scale=2, thickness=2, offset=10)
        cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=2, colorR=(100, 100, 255), colorC=(255, 255, 0))

        cx, cy = x1 + w // 2, y1 + h // 2
        # cv2.circle(img, (cx, cy), 5, (255, 0 , 0), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 30 < cy < limits[1] + 40:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (255, 255, 0), 2)

    cvzone.putTextRect(img, f' Counted Cars: {len(totalCount)}', (200, 700),
                       scale=2, thickness=2, offset=10,
                       colorR=(255, 0, 0))

    cv2.imshow("Car Counter", img)
    # cv2.imshow("Img Region", imgRegion)

    cv2.waitKey(1)

    #pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
