from ultralytics import YOLO
import cv2

import use
from sort.sort import *
from use import get_car, read_license_plate, write_csv

results = {}
tracker = Sort()
# load models
yolo_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('./cars.mp4')
vehicles = [2, 3, 5, 7] #id object in yolo model 2 - car, 3- motorbike, 5 - bus, 7 - truck
# read frames
frame_num = -1
ret = True
while ret:
    frame_num += 1
    ret, frame = cap.read()
    if ret and frame_num < 10: #only 10 cars
        results[frame_num] = {}
        detections = yolo_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        track_id = tracker.update(np.asarray(detections_)) #which car represent

        plates = license_plate_detector(frame)[0]
        for plate in plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = plate

            #which car belongs to which plate - method in use.py
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(plate, track_id)

            plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            _, plate_thresh = cv2.threshold(plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

            #read plate - method in use.py
            plate_text, plate_text_score = read_license_plate(plate_thresh)

            #for every car and plate save information
            if plate_text is not None:
                results[frame_num][car_id] = {'car':{'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                              'plate': {'bbox': [x1, y1, x2, y2],
                                              'text': plate_text,
                                              'bbox_score': score,
                                              'text_score': plate_text_score}}

write_csv(results, "./result.csv")
