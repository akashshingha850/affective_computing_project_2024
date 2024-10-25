#!/bin/bash

# Function to handle cleanup on exit
cleanup() {
    echo "Stopping the script..."
    exit 0
}

# Trap signals to allow cleanup
trap cleanup SIGINT SIGTERM

# Infinite loop to keep the script running
while true; do
    /bin/python3 /home/jetson/affective_computing/run.py
done
