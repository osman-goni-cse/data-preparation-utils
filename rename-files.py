import os
import shutil
from tqdm import tqdm

def rename_images(input_folder, output_folder, prefix="image_"):
    """
    Renames image files in the input folder and saves them in the output folder.

    Args:
        input_folder (str): Path to the folder containing the original images.
        output_folder (str): Path to the folder where renamed images will be saved.
        prefix (str): Prefix for the renamed files (default is "image_").
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the input folder
    files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])

    # Rename and copy each file
    for i, filename in enumerate(tqdm(files, desc="Renaming files", unit="file"), start=1185):
        # Generate the new filename
        new_filename = f"{prefix}{i:04d}.jpg"  # e.g., image_0001.jpg, image_0002.jpg

        # Full paths for the input and output files
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, new_filename)

        # Copy and rename the file
        shutil.copy(input_path, output_path)

# Example usage
input_folder = "/home/osman/Downloads/Dataset/ContainerNum_dataset/train/images"  # Replace with the path to your input folder
output_folder = "/home/osman/Downloads/Dataset/CCMS/renamed_ContainerNum_dataset"
rename_images(input_folder, output_folder, prefix="container_")
