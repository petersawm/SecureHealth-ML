#!/bin/bash

# SecureHealth-ML Setup Script
# This script sets up the project environment

echo "=================================================="
echo "SecureHealth-ML Setup"
echo "=================================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if Python 3.9+ is installed
required_version="3.9"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "Error: Python 3.9 or higher is required"
    exit 1
fi

echo "✓ Python version is compatible"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi

echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

echo "✓ Pip upgraded"
echo ""

# Install requirements
echo "Installing dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo "✓ Dependencies installed"
echo ""

# Test imports
echo "Testing installation..."
python3 << EOF
try:
    import torch
    import flwr
    import opacus
    import numpy
    import sklearn
    print("✓ All core packages imported successfully")
except ImportError as e:
    print(f"Error: Failed to import package: {e}")
    exit(1)
EOF

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "To get started:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run the simulation:"
echo "     python simulation.py"
echo ""
echo "  Or start server and clients manually:"
echo "     Terminal 1: python server.py"
echo "     Terminal 2: python client.py --client-id 0"
echo "     Terminal 3: python client.py --client-id 1"
echo ""
echo "For more information, see README.md and QUICKSTART.md"
echo "=================================================="
