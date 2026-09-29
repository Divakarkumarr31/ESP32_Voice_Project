#!/bin/bash

set -e

echo "========================================"
echo "ARISE Project Setup"
echo "========================================"

echo ""
echo "Creating Python virtual environment..."

cd "$(dirname "$0")/../server"

python3 -m venv .venv

echo ""
echo "Activating virtual environment..."

source .venv/bin/activate

echo ""
echo "Installing Python dependencies..."

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"

echo ""
echo "To start the ARISE server:"
echo ""
echo "cd server"
echo "source .venv/bin/activate"
echo "python arise_server.py"
echo ""