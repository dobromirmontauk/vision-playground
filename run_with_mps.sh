#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Set environment variables to ensure PyTorch uses MPS
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run the application with the CLIP detector on MPS (Apple Silicon GPU)
echo "Starting application with CLIP detector on Apple Silicon GPU..."
echo "Using 1000 household items categories from data file..."
python app.py --model clip --max-categories 1000 > server.log 2>&1 &

# Store the PID
echo $! > server.pid
echo "Server started with PID $(cat server.pid). Logs in server.log"
echo "View the application at http://localhost:5000"
echo "Press Ctrl+C to stop viewing logs, or run ./stop_server.sh to stop the server"

# Show logs to help debug any MPS-related issues
echo "Showing server logs (Ctrl+C to exit log view):"
tail -f server.log