import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ================================ DATASET STRUCTURE ================================
# Define paths
IMAGE_DIR = "data/images"     # Folder containing image files
LABEL_DIR = "data/labels"     # Folder containing annotation files (.txt)
CSV_DIR = "data/CSVs"         # Folder to save dataset CSV

# Make sure the CSV folder exists
if not os.path.exists(CSV_DIR):
    os.makedirs(CSV_DIR)

# Step 1: Initialize empty DataFrame
data_df = pd.DataFrame(columns=["images", "labels"])

# Step 2: Get list of all image files
all_images = []
for file_name in os.listdir(IMAGE_DIR):
    # Check if the file ends with .jpg, .jpeg, or .png (case insensitive)
    if file_name.lower().endswith(".jpg") or file_name.lower().endswith(".jpeg") or file_name.lower().endswith(".png"):
        all_images.append(file_name)

# Step 3: Loop through each image and find corresponding label
for img_file in sorted(all_images):
    # Extract file ID without extension
    file_id, extension = os.path.splitext(img_file)

    # Construct the expected label file path
    label_file = file_id + ".txt"
    label_path = os.path.join(LABEL_DIR, label_file)

    # Check if the corresponding label exists
    if not os.path.exists(label_path):
        print("WARNING: Label file for image '{}' not found. This image will be skipped.".format(img_file))
        continue

    # Construct full image path
    image_path = os.path.join(IMAGE_DIR, img_file)

    # Create a new row as a dictionary
    row = {
        "images": image_path,
        "labels": label_path
    }

    # Append the row to the DataFrame
    data_df = pd.concat([data_df, pd.DataFrame([row])], ignore_index=True)

# Step 4: Save DataFrame to CSV
csv_file_path = os.path.join(CSV_DIR, "dataset.csv")
data_df.to_csv(csv_file_path, index=False)


# Step 5: Print full DataFrame
# ================================
# Ensure pandas will display all rows
pd.set_option("display.max_rows", None)

print("Dataset CSV created successfully!")
print(data_df)


# ================================ DATA SPLITTING ================================

dataset = pd.read_csv(r"data/CSVs/dataset.csv")

train_data, val_data = train_test_split(dataset, test_size=0.3)
train_data.to_csv(r"data/CSVs/train_df.csv", index=False)
val_data.to_csv(r"data/CSVs/val_df.csv", index=False)
