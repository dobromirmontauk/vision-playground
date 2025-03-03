#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Run the application with the CLIP detector and household item categories
echo "Starting application with CLIP detector and 1000 household item categories..."
python app.py --model clip --max-categories 1000 > server.log 2>&1 &

# Store the PID
echo $! > server.pid
echo "Server started with PID $(cat server.pid). Logs in server.log"
echo "View the application at http://localhost:5000"
echo "Using household items categories from data/categories/household_items.txt"