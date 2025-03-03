#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Run the application with the CLIP detector
echo "Starting application with CLIP object detector..."
python app.py --model clip > server.log 2>&1 &

# Store the PID
echo $! > server.pid
echo "Server started with PID $(cat server.pid). Logs in server.log"
echo "View the application at http://localhost:5000"