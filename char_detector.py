import os
import time
from ultralytics import YOLO
import numpy as np
import cv2

__dir__ = os.path.dirname(os.path.abspath(__file__))
model_relative_path = 'models/yolov8_char_det/best_small.pt'
model_path = os.path.normpath(os.path.join(__dir__, model_relative_path))

confidence_threshold = 0.5

# Yolo8 Character Detector for container number
class CharDetector:
    def __init__(self):
        self.model = YOLO(model_path)
        self.warmup()
        print("CharDetector loaded and warmed up successfully.")

    def warmup(self):
        _dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(_dummy_image, verbose=False)
    
    def detect(self, image):
        # cv2.imwrite("temp_images/cropped.jpg", image)
        height, width, _ = image.shape
        is_vertical = False
        if height > width:
            print("input cn cropped image is vertical.")
            is_vertical = True
        else:
            print("input cn cropped image is horizontal.")

        st = time.time()
        res = self.model(image, verbose=False)
        boxes = res[0].boxes.numpy().data
        # boxes: [[x1, y1, x2, y2, conf, cls],[...box2....],[...box3...],..., [...boxN...]]]
        num_boxes = len(boxes)
        print(f"{num_boxes} char boxes detected. Took {time.time() - st:.3f} seconds.")
        #print('All chars det confidence:', [box[4] for box in boxes])

        new_boxes = [box for box in boxes if box[4] >= confidence_threshold]

        if num_boxes - len(new_boxes) > 0:
            print(f"{num_boxes - len(new_boxes)} char boxes removed, confidence below {confidence_threshold}.")
        else:
            print(f"All char boxes kept, confidence above {confidence_threshold}.")
        
        debug_image = image.copy()
        for box in new_boxes:
            x1, y1, x2, y2, conf, cls = box
            cv2.rectangle(debug_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        # cv2.imwrite("temp_images/chars_debug.jpg", debug_image)
        
        is_reassembled = False

        if len(new_boxes) >= 11 and is_vertical:
            new_boxes = self.reorder_boxes(is_vertical, new_boxes)
            image = self.reassemble_characters(image, is_vertical, new_boxes)
            is_reassembled = True
            print(f"Reassembled {len(new_boxes)} char boxes into a new image.")
        elif len(new_boxes) == 11 and not is_vertical:
            new_boxes = self.reorder_boxes(is_vertical, new_boxes)
            image = self.reassemble_characters(image, is_vertical, new_boxes)
            is_reassembled = True
            print(f"Reassembled {len(new_boxes)} char boxes into a new image.")
        elif len(new_boxes) > 11 and not is_vertical:
            is_reassembled = False
            print(f"More than 11 char boxes detected, but horizontal, return original cropped image")
        else:
            is_reassembled = False
            print(f"Only {len(new_boxes)} char boxes left, return original cropped image")
        
        # cv2.imwrite("temp_images/chars.jpg", image)
        
        return image, is_vertical, is_reassembled
    
    def reassemble_characters(self, image, is_vertical, boxes):
        # define the expansion size of character region
        if is_vertical:
            expand_x, expand_y = 10, 2
        else:
            expand_x, expand_y = 2, 5

        # store the cropped and adjusted character regions
        cropped_images = []

        # iterate over each bounding box, crop and expand the character region   
        for box in boxes:
            x1, y1, x2, y2 = box[:4]

            # calculate the expanded coordinates, make sure not to exceed the image boundary
            x1_expanded = max(x1 - expand_x, 0)
            y1_expanded = max(y1 - expand_y, 0)
            x2_expanded = min(x2 + expand_x, image.shape[1])
            y2_expanded = min(y2 + expand_y, image.shape[0])

            # crop the expanded region
            cropped = image[int(y1_expanded):int(y2_expanded), int(x1_expanded):int(x2_expanded)]
            cropped_images.append(cropped)

        # calculate the max height of all cropped regions
        max_height = max([img.shape[0] for img in cropped_images])

        # add black padding to adjust the height of each region
        adjusted_images = []
        for img in cropped_images:
            # calculate the padding size
            padding_top = (max_height - img.shape[0]) // 2
            padding_bottom = max_height - img.shape[0] - padding_top

            # add padding
            padded = cv2.copyMakeBorder(img, padding_top, padding_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            adjusted_images.append(padded)

        # horizontal concatenation of character regions
        new_image = np.concatenate(adjusted_images, axis=1)

        # add extra black padding to the top and bottom of the new image
        padding_height = 3
        new_image_padded = cv2.copyMakeBorder(new_image, padding_height, padding_height, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return new_image_padded

    def reorder_boxes(self, is_vertical ,boxes):
        if is_vertical:
            # sort by y1 from top to bottom
            boxes = sorted(boxes, key=lambda box: box[1])
        else:
            # sort by x1 from left to right
            boxes = sorted(boxes, key=lambda box: box[0])

        return boxes

"""
detector = CharDetector()
test_image = cv2.imread("test_images/crop_1.jpg")
result = detector.detect(test_image)
print(result)
"""