import xml.etree.ElementTree as ET
import os
from tqdm import tqdm # for progress bar

# define the paths for training set
raw_cvat_annotation_file = '/home/user/project/data/cvat_annotations/container_annotation_for_train.xml'
output_labels_folder = '/home/user/project/data/yolov8_det_data/train/labels'
# do the same for the validation set
# raw_cvat_annotation_file = '/home/user/project/data/cvat_annotations/container_annotation_for_valid.xml'
# output_labels_folder = '/home/user/project/data/yolov8_det_data/valid/labels'

# Define the interested labels, what we want to detect
interested_labels = {'CN': 0, 'CN_ABC': 1, 'CN_NUM': 2, 'TS': 3}

if not os.path.exists(output_labels_folder):
    os.makedirs(output_labels_folder)

tree = ET.parse(raw_cvat_annotations_file)
root = tree.getroot()

images = root.findall('image')

for image in tqdm(images, desc="Converting to YOLO labels"):
    image_file = image.get('name')
    image_width = int(image.get('width'))
    image_height = int(image.get('height'))
    
    label_file = os.path.join(output_labels_folder, os.path.splitext(image_file)[0] + '.txt')
    with open(label_file, 'w') as file:
        for box in image.findall('box'):
            label = box.get('label')
            if label in interested_labels:
                # normalize the coordinates
                xtl = float(box.get('xtl')) / image_width
                ytl = float(box.get('ytl')) / image_height
                xbr = float(box.get('xbr')) / image_width
                ybr = float(box.get('ybr')) / image_height
                # convert to yolo format
                x_center = (xtl + xbr) / 2
                y_center = (ytl + ybr) / 2
                width = xbr - xtl
                height = ybr - ytl

                file.write(f"{interested_labels[label]} {x_center} {y_center} {width} {height}\n")
