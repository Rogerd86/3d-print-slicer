#!/bin/bash
echo "============================================"
echo "  3D Print Slicer - Install and Run"
echo "============================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+"
    exit 1
fi

echo "Installing dependencies..."
pip3 install -r requirements.txt

echo "Launching 3D Print Slicer..."
python3 main.py
