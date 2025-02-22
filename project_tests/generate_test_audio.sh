#!/bin/bash

# Need to install ffmpeg for these to work

# Check if ffmpeg and sox are installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg not found. Install it with 'brew install ffmpeg'"
    exit 1
fi

if ! command -v sox &> /dev/null; then
    echo "Error: sox not found. Install it with 'brew install sox'"
    exit 1
fi

# Create an output directory
mkdir -p test_audio

# Generate a base sine wave file (5s, 440Hz, normalized volume)
ffmpeg -f lavfi -i "sine=frequency=440:duration=5" -c:a pcm_s16le test_audio/base.wav

# 1. Over-Amplified/Clipped Audio
ffmpeg -i test_audio/base.wav -filter:a "volume=10" test_audio/clipped.wav

# 2. Very Quiet Audio
ffmpeg -i test_audio/base.wav -filter:a "volume=0.05" test_audio/quiet.wav

# 3. Uneven Volume Levels
ffmpeg -i test_audio/base.wav -af "volume='if(lt(t,2),0.1,if(lt(t,4),2,1))'" test_audio/uneven.wav

# 4. Background Noise
ffmpeg -i test_audio/base.wav -filter_complex "anullsrc=r=44100:cl=2 [noise]; [0:a][noise] amix=inputs=2:duration=first:weights=1 0.5" test_audio/noisy.wav

# 5. Synthetic Over-Amplified Test (1kHz tone)
sox -n test_audio/overload.wav synth 5 sine 1000 vol 2

echo "Test audio files generated in 'test_audio' directory."
