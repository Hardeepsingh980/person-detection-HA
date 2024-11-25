#!/bin/bash
echo "Starting Person Detection Add-on..."

# Ensure the TensorFlow Lite model exists in the working directory
if [ ! -f "1.tflite" ]; then
    echo "TensorFlow Lite model (1.tflite) not found. Please add it to the add-on folder."
    exit 1
fi

# Run the Python script
python3 detection.py
