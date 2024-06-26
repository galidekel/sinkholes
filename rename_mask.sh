DIRECTORY="/home/labs/rudich/Rudich_Collaboration/deadsea_sinkholes_data/patches/mask_patches_H200_W100_strpp2_11days"


# Loop through each file in the directory
for FILE in "$DIRECTORY"/*; do
  # Ensure we are dealing with a file
  if [ -f "$FILE" ]; then
    # Get the filename without the directory path
    FILENAME=$(basename "$FILE")

    # Check if the filename matches the pattern
    if [[ "$FILENAME" == mask_patches_*_H200_W100.npy ]]; then
      # Add _strpp2 before the .npy extension
      NEW_FILENAME="${FILENAME%.npy}_strpp2.npy"

      # Move (rename) the file
      mv "$FILE" "$DIRECTORY/$NEW_FILENAME"

      # Output the rename action
      echo "Renamed $FILENAME to $NEW_FILENAME"
    fi
  fi
done
