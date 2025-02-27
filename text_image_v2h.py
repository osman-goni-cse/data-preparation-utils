# step 2: text_image_v2h.py
# How to use: python3 text_image_v2h.py
import cv2 
import os
from v2h_char_detector import V2HCharDetector # character detection, vertical to horizontal

# cropped the container number images from the original container images
cropped_images_folder = '/home/user/project/data/cropped_container_number_images'
# the folder to save the updated cropped images, include originally horizontal images and rearranged horizontal images
updated_cropped_images_folder = '/home/user/project/data/updated_cropped_images'
# the images that are not processed will be saved in the need_manual_check_images_folder
need_manual_check_images_folder = '/home/user/project/data/need_manual_check_images'

if not os.path.exists(updated_cropped_images_folder):
    os.makedirs(updated_cropped_images_folder)
if not os.path.exists(need_manual_check_images_folder):
    os.makedirs(need_manual_check_images_folder)

v2h_char_detector = V2HCharDetector() # it will return the rearranged horizontal image

processed_count = 0

for filename in os.listdir(cropped_images_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(cropped_images_folder, filename)
        img = cv2.imread(image_path)

        if img is not None:
            height, width = img.shape[:2]
            # check if image is vertical
            if height > width:
                # vertical image
                print(filename + " is vertical)
                # detect the characters and rearrange them horizontally, return the rearranged image
                is_res_ok, processed_img = v2h_char_detector.detect(img)
                processed_count += 1
                if is_res_ok:
                    cv2.imwrite(os.path.join(updated_cropped_images_folder , filename), processed_img)
                else:
                    # if the char detection is not ok, save the image to the need_manual_check_images_folder
                    cv2.imwrite(os.path.join(need_manual_check_images_folder, filename), processed_img)
                    print(filename, " is not processed, please check manually.")
            else:
                # horizontal image, no need to process
                cv2.imwrite(os.path.join(updated_cropped_images_folder, filename), img)

print(processed_count, " images processed.")
