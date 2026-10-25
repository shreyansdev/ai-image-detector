#!/bin/bash
# Startup script for Linux and macOS

# Ensure we are in the script's directory
cd "$(dirname "$0")"

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH."
    echo "Please install Python 3.10+ to run this project."
    read -p "Press enter to exit..."
    exit 1
fi

python3 run.py
