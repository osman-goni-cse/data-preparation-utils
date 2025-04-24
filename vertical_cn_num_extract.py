# step 1: cvat_to_pdlocrrec_label.py
# How to use: python3 cvat_to_pdlocrrec_label.py
import xml.etree.ElementTree as ET
import cv2 # for cropping images
import os
from tqdm import tqdm # for progress bar

raw_images_folder = '/home/osman/Downloads/Dataset/CCMS'
raw_cvat_annotation_file = '/home/osman/Downloads/Dataset/CCMS/annotations.xml'
cropped_images_folder = '/home/osman/Downloads/Dataset/CCMS/cropped_vertical_container_number_images'
cropped_labels_file = '/home/osman/Downloads/export-data/cropped_vertical_container_number_labels.txt'

tree = ET.parse(raw_cvat_annotation_file)
root = tree.getroot()

if not os.path.exists(cropped_images_folder):
    os.makedirs(cropped_images_folder)

# calculate total CN labels count
total_cn_count = sum(1 for image in root.findall('.//image') for box in image.findall('.//box') if box.get('label') == 'CN')

with open(cropped_labels_file, 'w') as file, tqdm(total=total_cn_count, desc="Processing CN labels for PaddleOCR Rec") as pbar:
    for image in root.findall('.//image'):
        image_name = image.get('name')
        base_name = os.path.splitext(image_name)[0]
        cn_count = 1

        for box in image.findall('.//box'):  # one image may have multiple CN
            if box.get('label') == 'CN':
                cn_text = box.find(".//attribute[@name='cn_text']").text
                new_image_name = f"{base_name}_{cn_count:02}.jpg"
                xtl, ytl, xbr, ybr = map(lambda x: round(float(box.get(x))), ['xtl', 'ytl', 'xbr', 'ybr'])

                object_width = abs(xbr - xtl)
                object_height = abs(ytl - ybr)

                if object_height >= object_width:
                    file.write(f"{new_image_name}\t{cn_text}\n")

                    img = cv2.imread(os.path.join(raw_images_folder, image_name))
                    cropped_img = img[ytl:ybr, xtl:xbr]
                    # print(f"Cropping image: {image_name}, CN: {cn_count}, Coordinates: ({xtl}, {ytl}), ({xbr}, {ybr})")
                    # print(f"{os.path.join(cropped_images_folder, new_image_name)}")
                    cv2.imwrite(os.path.join(cropped_images_folder, new_image_name), cropped_img)

                    cn_count += 1
                    pbar.update(1)

print(f"Total CN labels processed: {total_cn_count}")
