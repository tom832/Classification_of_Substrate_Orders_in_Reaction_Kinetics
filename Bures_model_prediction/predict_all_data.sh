#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
shopt -s nullglob

# Directory containing the CSV files
DATA_DIR="Data/all_data/"
MODEL_FILE="Data/M1_20_model_S_noXS_01to5.h5"
csv_files=("$DATA_DIR"*.csv)
if (( ${#csv_files[@]} == 0 )); then
  echo 'No input CSVs. Run the data-generation cells in generate_predict_analyze_data.ipynb first.' >&2
  exit 1
fi
mkdir -p all_data_output all_data_plot

# Loop through all .csv files in the directory
for csv_file in "${csv_files[@]}"
do
  # Extract the filename without the path and extension
  filename_with_ext=$(basename "$csv_file")
  filename_no_ext="${filename_with_ext%.csv}"

  echo "Processing $filename_with_ext..."

  # Run your Python script
  "${PYTHON:-python}" predict.py "$csv_file" --model "$MODEL_FILE" --plot_name "$filename_no_ext"

  echo "Finished processing $filename_with_ext."
  echo "---" # Separator for clarity
done

echo "All files processed."
