#!/bin/bash
source venv/bin/activate
python app.py > server.log 2>&1 &
echo $! > server.pid
echo "Server started with PID $(cat server.pid). Logs in server.log"