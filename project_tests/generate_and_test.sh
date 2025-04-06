#!/bin/bash

# Test script used to measure timing performances from various length files

# --- Config ---
OUT_DIR="test_audio"
LOG_FILE="${OUT_DIR}/audio_tests.log"  # Log file path
SAMPLE_RATE=44100
FREQ=440
AUDIO_NORMALIZER_PATH="./audio_normalizer" # change this with your absolute path to executable

# Ensure SoX is installed
if ! command -v sox &> /dev/null; then
    echo "Error: SoX is not installed. Please install it and try again." | tee -a "$LOG_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUT_DIR"

# Clear or create log file
echo "=== Audio Test Log - $(date) ===" > "$LOG_FILE"

# Durations to test (in minutes)
DURATIONS=(0.1 1 10 20)

# Generate and process files
echo "Generating and testing .wav files..."
echo "Generating and testing .wav files..." >> "$LOG_FILE"

for DURATION_MIN in "${DURATIONS[@]}"; do
    # Convert minutes to seconds for SoX
    DURATION_SEC=$(awk "BEGIN {print $DURATION_MIN * 60}")
    DURATION_SAFE=$(echo "$DURATION_MIN" | tr '.' '_')

    # Generate filenames
    MONO_8BIT="${OUT_DIR}/mono_8bit_${DURATION_SAFE}min.wav"
    STEREO_8BIT="${OUT_DIR}/stereo_8bit_${DURATION_SAFE}min.wav"
    MONO_16BIT="${OUT_DIR}/mono_16bit_${DURATION_SAFE}min.wav"
    STEREO_16BIT="${OUT_DIR}/stereo_16bit_${DURATION_SAFE}min.wav"

    # Generate files with SoX
    echo "Generating ${DURATION_SAFE}min files..."
    echo "Generating ${DURATION_SAFE}min files..." >> "$LOG_FILE"
    sox -n -r $SAMPLE_RATE -c 1 -b 8 "$MONO_8BIT" synth $DURATION_SEC sine $FREQ gain -3 >> "$LOG_FILE" 2>&1
    sox -n -r $SAMPLE_RATE -c 2 -b 8 "$STEREO_8BIT" synth $DURATION_SEC sine $FREQ gain -3 >> "$LOG_FILE" 2>&1
    sox -n -r $SAMPLE_RATE -c 1 -b 16 "$MONO_16BIT" synth $DURATION_SEC sine $FREQ gain -3 >> "$LOG_FILE" 2>&1
    sox -n -r $SAMPLE_RATE -c 2 -b 16 "$STEREO_16BIT" synth $DURATION_SEC sine $FREQ gain -3 >> "$LOG_FILE" 2>&1

    # Run audio_normalizer and log output
    echo "Processing ${DURATION_MIN}min files..."
    echo "Processing ${DURATION_MIN}min files..." >> "$LOG_FILE"
    echo "Processing ${DURATION_MIN} MONO_8BIT"
    echo "Processing ${DURATION_MIN} MONO_8BIT" >> "$LOG_FILE"
    "$AUDIO_NORMALIZER_PATH" "$MONO_8BIT" -o "${OUT_DIR}/output_8bit_mono_${DURATION_SAFE}min.wav" -n "8bit_mono_${DURATION_SAFE}min" >> "$LOG_FILE" 2>&1
    echo "Processing ${DURATION_MIN} STEREO_8BIT"
    echo "Processing ${DURATION_MIN} STEREO_8BIT" >> "$LOG_FILE"
    "$AUDIO_NORMALIZER_PATH" "$STEREO_8BIT" -o "${OUT_DIR}/output_8bit_stereo_${DURATION_SAFE}min.wav" -n "8bit_stereo_${DURATION_SAFE}min" >> "$LOG_FILE" 2>&1
    echo "Processing ${DURATION_MIN} MONO_16BIT"
    echo "Processing ${DURATION_MIN} MONO_16BIT" >> "$LOG_FILE"
    "$AUDIO_NORMALIZER_PATH" "$MONO_16BIT" -o "${OUT_DIR}/output_16bit_mono_${DURATION_SAFE}min.wav" -n "16bit_mono_${DURATION_SAFE}min" >> "$LOG_FILE" 2>&1
    echo "Processing ${DURATION_MIN} STEREO_16BIT"
    echo "Processing ${DURATION_MIN} STEREO_16BIT" >> "$LOG_FILE"
    "$AUDIO_NORMALIZER_PATH" "$STEREO_16BIT" -o "${OUT_DIR}/output_16bit_stereo_${DURATION_SAFE}min.wav" -n "16bit_stereo_${DURATION_SAFE}min" >> "$LOG_FILE" 2>&1
done

echo "All files generated and processed in $OUT_DIR:"
echo "All files generated and processed in $OUT_DIR:" >> "$LOG_FILE"
ls -lh "$OUT_DIR" >> "$LOG_FILE"

echo "Done! Log saved to $LOG_FILE"