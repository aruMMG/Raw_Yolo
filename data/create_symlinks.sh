#!/bin/bash

# Define the root source folder
SOURCE_DIR="/media/singh_a/Seagate Expansion Drive/RGBTHDRDataset/Generated_IR/SDR2IR_UNet_baseline/results/RGBTHDRDataset_test"

# Define the destination folder
DEST_DIR="/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/side_work/yolov9/data/RGBTHDRDataset_test/images/val"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Loop through each .tiff file in the source directory
find "$SOURCE_DIR" -maxdepth 1 -type f -iname "RGB*.png" | while read -r file; do
  filename=$(basename "$file")
  
  # Extract numeric part from filename (e.g., T00001.tiff -> 00001)
  number=$(echo "$filename" | grep -oP '(?<=RGB)\d{5}(?=\.png)')
  
  # Add 2 to the number and keep leading zeros (5 digits)
  new_number=$(printf "%05d" $((10#$number + 3)))  # 10# prevents octal issues

  # Construct new filename
  new_filename="T${new_number}.png"

  # Create symlink
  ln -s "$file" "$DEST_DIR/$new_filename"
done

echo "Symlinks with updated filenames created."
