#!/bin/bash

# Quick start script for AgamScan API

echo "================================"
echo "AgamScan API - Quick Start"
echo "================================"

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo "📝 Creating .env from template..."
    cp .env.example .env
    echo "✅ .env created. Please edit it with your credentials:"
    echo "   nano .env"
    echo ""
    echo "Required values:"
    echo "  - AZURE_VISION_KEY"
    echo "  - AZURE_VISION_ENDPOINT"
    echo "  - AZURE_AI_KEY"
    echo "  - AZURE_AI_ENDPOINT"
    echo "  - YOLO_MODEL_PATH"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
if [ ! -f "venv/.installed" ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
    touch venv/.installed
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

# Start the API
echo ""
echo "🚀 Starting AgamScan API..."
echo "📖 Swagger docs will be available at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
