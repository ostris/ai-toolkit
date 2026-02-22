#!/bin/bash

# JoyCaption Service Startup Script
# This script starts the JoyCaption captioning service

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default configuration
HOST="${CAPTION_HOST:-127.0.0.1}"
PORT="${CAPTION_PORT:-5000}"
MODEL="${CAPTION_MODEL:-fancyfeast/llama-joycaption-beta-one-hf-llava}"

echo "Starting JoyCaption service..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Model: $MODEL"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Start the service
echo "Starting caption server..."
python caption_server.py --host "$HOST" --port "$PORT" --model "$MODEL" --preload
