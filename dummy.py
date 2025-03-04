import xml.etree.ElementTree as ET
import os
from tqdm import tqdm # for progress bar
import cv2
import numpy as np

# define the paths for training set
raw_cvat_annotation_file = '/home/osman/Videos/annotations.xml'
# output_labels_folder = '/home/user/project/data/yolov8_det_data/train/labels'
# do the same for the validation set
# raw_cvat_annotation_file = '/home/user/project/data/cvat_annotations/container_annotation_for_valid.xml'
# output_labels_folder = '/home/user/project/data/yolov8_det_data/valid/labels'

# Define the interested labels, what we want to detect
# interested_labels = {'CN': 0, 'CN_ABC': 1, 'CN_NUM': 2, 'TS': 3}

# if not os.path.exists(output_labels_folder):
#     os.makedirs(output_labels_folder)

def rearrange_vertical_to_horizontal(image, bounding_boxes):
    # Sort bounding boxes by y-coordinate to maintain the vertical order
    bounding_boxes = sorted(bounding_boxes, key=lambda box: int(box[1]))
    
    # Extract individual character images
    char_images = [image[int(ytl):int(ybr), int(xtl):int(xbr)] for (xtl, ytl, xbr, ybr) in bounding_boxes]
    
    # Determine new canvas size
    total_width = sum(int(xbr) - int(xtl) for (xtl, ytl, xbr, ybr) in bounding_boxes)
    max_height = max(int(ybr) - int(ytl) for (xtl, ytl, xbr, ybr) in bounding_boxes)
    
    # Create a blank white canvas
    new_image = 255 * np.ones((max_height, total_width, 3), dtype=np.uint8)
    
    # Place each character horizontally
    x_offset = 0
    for char_img in char_images:
        h, w = char_img.shape[:2]
        new_image[0:h, x_offset:x_offset+w] = char_img
        x_offset += w
    
    return new_image

def reassemble_characters(image, is_vertical, boxes):
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

def reorder_boxes(is_vertical ,boxes):
        if is_vertical:
            # sort by y1 from top to bottom
            boxes = sorted(boxes, key=lambda box: box[1])
        else:
            # sort by x1 from left to right
            boxes = sorted(boxes, key=lambda box: box[0])

        return boxes


tree = ET.parse(raw_cvat_annotation_file)
root = tree.getroot()

images = root.findall('image')


for image in tqdm(images, desc="Converting to YOLO labels"):
    image_file = image.get('name')
    image_width = int(image.get('width'))
    image_height = int(image.get('height'))

    bounding_boxes = []

    
    for box in image.findall('box'):
        label = box.get('label')
        xtl = float(box.get('xtl')) 
        ytl = float(box.get('ytl')) 
        xbr = float(box.get('xbr')) 
        ybr = float(box.get('ybr')) 

        # print(f'{xtl} {ytl} {xbr} {ybr}')
        bounding_boxes.append((xtl, ytl, xbr, ybr))
    
    image = cv2.imread("/home/osman/Downloads/export-data/cropped_container_number_images/container-39_02.jpg")
    # output_image = rearrange_vertical_to_horizontal(image, bounding_boxes)
    new_boxes = reorder_boxes(True, bounding_boxes)
    output_image = reassemble_characters(image, True, new_boxes)
    cv2.imwrite("/home/osman/output1.png", output_image)
    break