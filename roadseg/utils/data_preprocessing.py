from PIL import Image
import os


# this function removes the microsoft logo in the bing data
def crop_last_24_lines(image_path, output_directory):
    # Open the image
    image = Image.open(image_path)

    # Get the image dimensions
    width, height = image.size

    # Calculate the coordinates for cropping
    left = 0
    upper = 0
    right = width
    lower = height - 24  # Crop the last 24 lines

    # Crop the image
    cropped_image = image.crop((left, upper, right, lower))

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Save the cropped image in the output directory
    file_name = os.path.basename(image_path)
    cropped_file_path = os.path.join(output_directory, "cropped_" + file_name)
    cropped_image.save(cropped_file_path)

# Specify the directory containing the images
input_directory ="/Users/selinbarash/Downloads/output2/label"

# Specify the directory to save the cropped images
output_directory = "/Users/selinbarash/Downloads/output2/cropped_label"

# Iterate over the files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Construct the full path to the image file
        image_path = os.path.join(input_directory, filename)

        # Call the function to crop the last 24 lines and save in the output directory
        crop_last_24_lines(image_path, output_directory)

def compare_directories(dir1, dir2):
    files1 = set(os.listdir(dir1))
    files2 = set(os.listdir(dir2))

    unique_files_dir1 = files1 - files2
    unique_files_dir2 = files2 - files1

    # Remove files unique to dir1
    for file in unique_files_dir1:
        file_path = os.path.join(dir1, file)
        os.remove(file_path)
        print("Removed", file, "from", dir1)

    # Remove files unique to dir2
    for file in unique_files_dir2:
        file_path = os.path.join(dir2, file)
        os.remove(file_path)
        print("Removed", file, "from", dir2)

# Specify the directories to compare
directory1 = "/Users/selinbarash/Downloads/output2/label"
directory2 = "/Users/selinbarash/Downloads/output2/sat"

# Call the function to compare the directories and remove the files
compare_directories(directory1, directory2)

import shutil

def calculate_white_pixel_percentage(image_path):
    # Open the image using PIL
    image = Image.open(image_path)
    # Convert the image to grayscale
    image = image.convert('L')

    # Get the size of the image
    width, height = image.size

    # Count the number of white pixels
    white_pixels = 0

    for x in range(width):
        for y in range(height):
            pixel_value = image.getpixel((x, y))
            if pixel_value == 255:  # Assuming white pixels have a value of 255
                white_pixels += 1

    # Calculate the percentage of white pixels
    total_pixels = width * height
    white_pixel_percentage = (white_pixels / total_pixels) * 100

    return white_pixel_percentage

# Define the input and output directories
input_directory = "/Users/selinbarash/Downloads/output2/cropped_label"
output_directory = "/Users/selinbarash/Downloads/output2/useless_images"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Iterate through the images in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Add any other image formats you want to process
        image_path = os.path.join(input_directory, filename)
        percentage = calculate_white_pixel_percentage(image_path)

        if percentage < 0.3:
            # Copy the image to the output directory
            output_path = os.path.join(output_directory, filename)
            shutil.copyfile(image_path, output_path)

            print(f"Image {filename} has {percentage}% white pixels and will be copied.")
        else:
            print(f"Image {filename} has {percentage}% white pixels and will not be copied.")


from PIL import Image
import numpy as np


def segment_map(image_path, save_path):
    # Open the image using PIL
    image = Image.open(image_path)
    # Convert the image to grayscale
    image = image.convert('L')

    # Convert the image to a NumPy array
    np_image = np.array(image)

    # Threshold the image to separate the road pixels from the rest
    threshold = 245  # Adjust this threshold value as per your requirements
    road_pixels = np_image > threshold
    segmented_image = np.zeros_like(np_image)
    segmented_image[road_pixels] = 255

    # Create a new PIL image from the segmented NumPy array
    segmented_image = Image.fromarray(segmented_image)

    # Save the segmented image
    segmented_image.save(save_path)


# Example usage
input_image_path = "/Users/selinbarash/Downloads/maps/testB/1_B.jpg"
output_image_path = "/Users/selinbarash/Downloads/maps/labels.jpg"

segment_map(input_image_path, output_image_path)



import random

def choose_random_elements(source_directory, destination_directory, num_elements=2000):
    # Get the list of files in the source directory
    file_list = os.listdir(source_directory)

    # Randomly select num_elements files
    selected_files = random.sample(file_list, num_elements)

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_directory, exist_ok=True)

    # Copy the selected files to the destination directory
    for file_name in selected_files:
        source_path = os.path.join(source_directory, file_name)
        destination_path = os.path.join(destination_directory, file_name)
        shutil.copy2(source_path, destination_path)

    print(f"Randomly selected {num_elements} elements and saved them in the destination directory.")


# Example usage
file_path = "/Users/selinbarash/Downloads/output2/useless_images"
output_directory = "/Users/selinbarash/Downloads/output2/random_useless_images"
num_elements = 2000

choose_random_elements(file_path, output_directory, num_elements)

# this function is to remove the elements specified in one directory from another one
def remove_duplicate_images(larger_directory, smaller_directory):
    # Get the list of files in the smaller directory
    smaller_files = os.listdir(smaller_directory)

    # Iterate over the files in the larger directory
    for file_name in os.listdir(larger_directory):
        file_path = os.path.join(larger_directory, file_name)

        # Check if the file is also present in the smaller directory
        if file_name in smaller_files:
            # Remove the file from the larger directory
            os.remove(file_path)
            print(f"Removed file: {file_name}")

    print("Duplicate images removed successfully.")

# Example usage
larger_directory = "/Users/selinbarash/Downloads/output2/cropped_label"
smaller_directory = "/Users/selinbarash/Downloads/output2/useless_images"

remove_duplicate_images(larger_directory, smaller_directory)

# this function is adds elements in one directory to another one
def add_elements(source_directory, destination_directory):
    # Get the list of files in the source directory
    file_list = os.listdir(source_directory)

    # Iterate over the files in the source directory
    for file_name in file_list:
        source_path = os.path.join(source_directory, file_name)
        destination_path = os.path.join(destination_directory, file_name)

        # Copy the file from source to destination
        shutil.copy(source_path, destination_path)

    print("Elements added successfully.")

# Example usage
source_directory = "/Users/selinbarash/Downloads/output2/sat_images_useless_images"
destination_directory = "/Users/selinbarash/Downloads/output2/cropped_sat"

add_elements(source_directory, destination_directory)

import os
import shutil

def copy_matching_files(larger_directory, smaller_directory, destination_directory):
    # Get the list of files in the smaller directory
    smaller_files = os.listdir(smaller_directory)

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_directory, exist_ok=True)

    # Iterate over the files in the larger directory
    for file_name in os.listdir(larger_directory):
        file_path = os.path.join(larger_directory, file_name)

        # Check if the file is also present in the smaller directory
        if file_name in smaller_files:
            # Copy the file to the destination directory
            destination_path = os.path.join(destination_directory, file_name)
            shutil.copy2(file_path, destination_path)
            print(f"Copied file: {file_name}")

    print("Matching files copied successfully.")

# Example usage
larger_directory = "/Users/selinbarash/Downloads/output2/cropped_sat copy"
smaller_directory = "/Users/selinbarash/Downloads/output2/random_useless_images"
destination_directory = "/Users/selinbarash/Downloads/output2/sat_images_useless_images"

copy_matching_files(larger_directory, smaller_directory, destination_directory)

import cv2
#
def apply_dilate(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Create a kernel for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Apply the dilate operation
    dilated_image = cv2.dilate(image, kernel, iterations=1)

    # Display the original and dilated images
    cv2.imshow('Original Image', image)
    cv2.imshow('Dilated Image', dilated_image)
    cv2.imwrite(output_image_path, dilated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = "/Users/selinbarash/Downloads/maps/dilate_3.jpg"
output_image_path = "/Users/selinbarash/Downloads/maps/dilate_4.jpg"
apply_dilate(image_path)


def apply_median_filter(input_image_path, output_image_path):
    # Load the input image using OpenCV
    input_image = cv2.imread(input_image_path)

    # Apply median filter with kernel size 3x3
    filtered_image = cv2.medianBlur(input_image, ksize=3)

    # Save the output image
    cv2.imwrite(output_image_path, filtered_image)

# Example usage
input_image_path = "/Users/selinbarash/Downloads/maps/vov.jpg"
output_image_path = "/Users/selinbarash/Downloads/maps/tol.jpg"
apply_median_filter(input_image_path, output_image_path)







