#!/bin/bash

# Directory containing WAV files
AUDIO_DIR="test_audio"

# Check if SoX is installed
if ! command -v sox &> /dev/null; then
    echo "Error: SoX is not installed. Please install it and try again."
    exit 1
fi

# Function to extract max value from WAV file
extract_max_sample() {
    local file="$1"

    # Get bit depth of the file
    BIT_DEPTH=$(sox --i -b "$file" 2>/dev/null)

    if [ "$BIT_DEPTH" == "8" ]; then
        echo "Processing 8-bit file: $file"
        # Convert to raw unsigned 8-bit data, extract max byte value
        MAX_VALUE=$(sox "$file" -t raw - | od -t u1 | awk '{for(i=2;i<=NF;i++) if ($i>max) max=$i} END {print max}')

    elif [ "$BIT_DEPTH" == "16" ]; then
        echo "Processing 16-bit file: $file"
        # Convert to raw signed 16-bit data, extract max sample value
        MAX_VALUE=$(sox "$file" -t raw - | od -t d2 | awk '{for(i=2;i<=NF;i++) if ($i>max) max=$i} END {print max}')

    else
        echo "Unsupported bit depth: $BIT_DEPTH for file $file"
        return
    fi

    echo "Max sample value in $file: $MAX_VALUE"
}

# Loop through all WAV files in the directory
for wav_file in "$AUDIO_DIR"/*.wav; do
    extract_max_sample "$wav_file"
done
