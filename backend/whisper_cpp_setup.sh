#!/bin/bash
# Script to set up Whisper.cpp for your Django application

# Create directory for Whisper.cpp
mkdir -p whisper_cpp
cd whisper_cpp

# Clone Whisper.cpp repository
git clone https://github.com/ggerganov/whisper.cpp.git .

# Build Whisper.cpp
make

# Download the base model (ggml-base.en.bin)
# This is ~148MB, relatively lightweight but good quality for English
bash ./models/download-ggml-model.sh base.en

# Return to the original directory
cd ..

echo "Whisper.cpp has been set up successfully!"
echo "Model downloaded: whisper_cpp/models/ggml-base.en.bin" 