#!/bin/bash

# Start script for AgamScan API and Admin Panel
# Runs both services in separate processes

echo "======================================"
echo "🚀 Starting AgamScan Services"
echo "======================================"

# Check if virtual environment exists
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    echo "⚠️  No virtual environment found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        source .venv/bin/activate
    fi
fi

# Create outputs directory if it doesn't exist
mkdir -p outputs/passed outputs/failed outputs/logs

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "======================================"
    echo "🛑 Stopping services..."
    echo "======================================"
    kill $API_PID 2>/dev/null
    kill $ADMIN_PID 2>/dev/null
    echo "✅ All services stopped"
    exit 0
}

# Trap Ctrl+C
trap cleanup INT TERM

# Start main API in background
echo ""
echo "📡 Starting Main API on http://localhost:8000"
python app.py > logs/api.log 2>&1 &
API_PID=$!
echo "   PID: $API_PID"

# Wait a moment for API to start
sleep 2

# Start admin panel in background
echo ""
echo "🎛️  Starting Admin Panel on http://localhost:5001"
python admin_panel.py > logs/admin.log 2>&1 &
ADMIN_PID=$!
echo "   PID: $ADMIN_PID"

echo ""
echo "======================================"
echo "✅ All services started successfully!"
echo "======================================"
echo ""
echo "📊 Access Points:"
echo "   Main API:      http://localhost:8000"
echo "   API Docs:      http://localhost:8000/docs"
echo "   Admin Panel:   http://localhost:5001"
echo ""
echo "📝 Logs:"
echo "   API Log:       logs/api.log"
echo "   Admin Log:     logs/admin.log"
echo ""
echo "💡 Tips:"
echo "   - View API docs: http://localhost:8000/docs"
echo "   - Monitor requests: http://localhost:5001"
echo "   - Press Ctrl+C to stop all services"
echo ""
echo "======================================"
echo "Waiting... (Press Ctrl+C to stop)"
echo "======================================"

# Wait for both processes
wait $API_PID $ADMIN_PID
