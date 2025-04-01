#!/bin/bash

# Define parameters
OUTPUT_FILE="ideal_test.wav"
SAMPLE_RATE=44100
DURATION=5  # in seconds
AMPLITUDE=0.2  # Set amplitude to a reasonable level for easy normalization detection

# Generate an ideal mono 8-bit PCM square wave
sox -b 8 -c 1 -r $SAMPLE_RATE -n $OUTPUT_FILE synth $DURATION square 440 vol $AMPLITUDE

# Verify the generated file
soxi $OUTPUT_FILE