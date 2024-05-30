#!/bin/bash

# Directory containing the files
DIRECTORY="/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/data_patches_H200_W100_strpp2_11days"

# String to be added
ADD_STRING="_strpp2"

# Position to insert the string (0-based index)
POSITION=40

# Loop through each file in the directory
for FILE in "$DIRECTORY"/*; do
  # Get the filename without the directory path
  FILENAME=$(basename "$FILE")

  # Get the first part of the filename
  PART1=${FILENAME:0:POSITION}

  # Get the second part of the filename
  PART2=${FILENAME:POSITION}

  # Construct the new filename
  NEW_FILENAME="${PART1}${ADD_STRING}${PART2}"

  # Rename the file
  mv "$FILE" "$DIRECTORY/$NEW_FILENAME"
done
