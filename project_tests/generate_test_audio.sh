#!/bin/bash

# Ensure SoX is installed
if ! command -v sox &> /dev/null
then
    echo "Error: SoX is not installed. Please install it and try again."
    exit 1
fi

# Output directory
OUT_DIR="test_audio"

# Settings
DURATION=600  # 10 minutes
SAMPLE_RATE=44100  # Standard CD quality sample rate

# Generate .wav files
echo "Generating test .wav files..."

sox -n -r $SAMPLE_RATE -c 1 -b 8 "./$OUT_DIR/mono_8bit_10min.wav" synth $DURATION pinknoise gain -3
sox -n -r $SAMPLE_RATE -c 2 -b 8 "./$OUT_DIR/stereo_8bit_10min.wav" synth $DURATION pinknoise gain -3
sox -n -r $SAMPLE_RATE -c 1 -b 16 "./$OUT_DIR/mono_16bit_10min.wav" synth $DURATION pinknoise gain -3
sox -n -r $SAMPLE_RATE -c 2 -b 16 "./$OUT_DIR/stereo_16bit_10min.wav" synth $DURATION pinknoise gain -3

echo "Files generated in $OUT_DIR:"
ls -lh "$OUT_DIR"

echo "Done!"