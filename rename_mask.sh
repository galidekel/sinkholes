#!/bin/bash

# Directory containing the files
DIRECTORY="/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/mask_patches_H200_W100_strpp2_11days"

# Loop through each file in the directory
for FILE in "$DIRECTORY"/*nonz*; do
  # Ensure we are dealing with a file
  if [ -f "$FILE" ]; then
    # Get the filename without the directory path
    FILENAME=$(basename "$FILE")

    # Check if the filename matches the old pattern
    if [[ "$FILENAME" == *"mask_patches_nonz_"*"_H200_W100.npy" ]]; then
      # Replace the old pattern with the new pattern
      NEW_FILENAME="${FILENAME/_H200_W100.npy/_H200_W100_strpp2.npy}"

      # Move (rename) the file
      mv "$FILE" "$DIRECTORY/$NEW_FILENAME"

      # Output the rename action
      echo "Renamed $FILENAME to $NEW_FILENAME"
    fi

     if [[ "$FILENAME" == *"mask_patches_2"*"_H200_W100.npy" ]]; then
      # Replace the old pattern with the new pattern
      NEW_FILENAME="${FILENAME/_H200_W100.npy/_H200_W100_strpp2.npy}"

      # Move (rename) the file
      mv "$FILE" "$DIRECTORY/$NEW_FILENAME"

      # Output the rename action
      echo "Renamed $FILENAME to $NEW_FILENAME"
    fi


  fi
done