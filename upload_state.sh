#!/bin/bash

dir="data/2018/1-Year"
mkdir -p "$dir"

for state in "$@"; do
  lower_state=$(echo "$state" | tr '[:upper:]' '[:lower:]')

  echo "Downloading data of $state..."
  url="https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_p${lower_state}.zip"
  zipfile="$dir/csv_p${lower_state}.zip"

  # Tenta baixar
  if wget "$url" -O "$zipfile"; then
    unzip -o "$zipfile" -d "$dir"
    rm -f "$zipfile"
  else
    echo "Error downloading data of $state."
  fi
done
