import os
import time
from ultralytics import YOLO
import numpy as np
import cv2

__dir__ = os.path.dirname(os.path.abspath(__file__))
model_relative_path = 'models/yolov8_cn_det/best_small.pt'
model_path = os.path.normpath(os.path.join(__dir__, model_relative_path))

confidence_threshold = 0.6

# Yolo8 Container-Number Detector (CN, CN_ABC, CN_NUM, TS)
class CNDetector:
    def __init__(self):
        self.model = YOLO(model_path)
        self.warmup()
        print("CNDetector loaded and warmed up successfully.")

    def warmup(self):
        _dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(_dummy_image, verbose=False)
    
    def detect(self, image):
        print("-----------------------------------------------")
        st = time.time()
        res = self.model(image, verbose=False)
        boxes = res[0].boxes.numpy().data
        # boxes: [[x1, y1, x2, y2, conf, class],[...box2....],[...box3...],..., [...boxN...]]]
        # class: 0 = CN, 1 = CN_ABC, 2 = CN_NUM, 3 = TS

        num_boxes = len(boxes)
        print(f"{num_boxes} CN/CN_ABC/CN_NUM/TS boxes detected. Took {time.time() - st:.3f} seconds.")

        new_boxes = []
        class_names ={0: "CN", 1: "CN_ABC", 2: "CN_NUM", 3: "TS"}

        for i in range(num_boxes):
            conf = boxes[i][4]
            class_id = boxes[i][5]
            if conf < confidence_threshold:
                print(f"{class_names[int(class_id)]} box confidence {conf:.3f} below {confidence_threshold}. Ignoring.")
            else:
                new_boxes.append(boxes[i])
                print(f"{class_names[int(class_id)]} box confidence {conf:.3f} above {confidence_threshold}. Keeping.")
        
        return new_boxes

"""
cn_detector = CNDetector()
test_image = cv2.imread("test_images/example.jpg")
result = cn_detector.detect(test_image)
print(result)
"""