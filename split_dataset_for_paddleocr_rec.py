# step 3: split_dataset_for_paddleocr_rec.py
# How to use: python3 split_dataset_for_paddleocr_rec.py
import os
import random
from shutil import copyfile
from tqdm import tqdm

images_path = '/home/user/project/data/updated_cropped_images'
labels_path = '/home/user/project/data/cropped_container_number_labels.txt'
train_images_path = '/home/user/project/data/paddleocr_rec_data/RecTrainData'
val_images_path = '/home/user/project/data/paddleocr_rec_data/RecEvalData'
train_labels_path = '/home/user/project/data/paddleocr_rec_data/rec_train_label.txt'
val_labels_path = '/home/user/project/data/paddleocr_rec_data/rec_eval_label.txt'

def split_dataset(total_images, train_ratio, images_folder, labels_file, train_images_folder, val_images_folder, train_labels_file, val_labels_file):
    with open(labels_file, 'r') as file:
        lines = file.readlines()

    total_images = min(total_images, len(lines))

    train_indices = set(random.sample(range(total_images), int(total_images * train_ratio)))

    if not os.path.exists(train_images_folder):
        os.makedirs(train_images_folder)
    if not os.path.exists(val_images_folder):
        os.makedirs(val_images_folder)

    with open(train_labels_file, 'w') as train_labels, open(val_labels_file, 'w') as val_labels:
        for i in tqdm(range(total_images), desc="Splitting dataset"):
            line = lines[i]
            image_name, label = line.split('\t')
            source_image_path = os.path.join(images_folder, image_name)

            if i in train_indices:
                copyfile(source_image_path, os.path.join(train_images_folder, image_name))
                train_labels.write(line)
            else:
                copyfile(source_image_path, os.path.join(val_images_folder, image_name))
                val_labels.write(line)

    print("Dataset split done.")

# change the total_images and train_ratio to fit your dataset
# train_ratio = 0.9 means 90% of the images will be used for training, 10% for validation
split_dataset(total_images=99999, train_ratio=0.9, images_folder=images_path, labels_file=labels_path, train_images_folder=train_images_path, val_images_folder=val_images_path, train_labels_file=train_labels_path, val_labels_file=val_labels_path)
