#!/bin/bash

# Directory containing the CSV files
DATA_DIR="Data/all_data/"
MODEL_FILE="Data/M1_20_model_S_noXS_01to5.h5"

# Loop through all .csv files in the directory
for csv_file in "$DATA_DIR"*.csv
do
  # Extract the filename without the path and extension
  filename_with_ext=$(basename "$csv_file")
  filename_no_ext="${filename_with_ext%.csv}"

  echo "Processing $filename_with_ext..."

  # Run your Python script
  python predict.py "$csv_file" --model "$MODEL_FILE" --plot_name "$filename_no_ext"

  echo "Finished processing $filename_with_ext."
  echo "---" # Separator for clarity
done

echo "All files processed."