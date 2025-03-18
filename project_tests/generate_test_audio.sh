#!/bin/bash

# Define parameters
OUTPUT_FILE="test.wav"
SAMPLE_RATE=44100
DURATION=1  # in seconds
AMPLITUDE=127  # Exact peak amplitude in unsigned 8-bit PCM (0-255 range)

# Generate a square wave with exact amplitude (forcing known peak values)
sox -b 8 -c 1 -r $SAMPLE_RATE -n $OUTPUT_FILE synth $DURATION square 440 vol $(bc -l <<< "$AMPLITUDE/255")

# Verify the generated file
soxi $OUTPUT_FILE
