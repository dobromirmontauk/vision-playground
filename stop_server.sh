#!/bin/bash

# Function to kill all related processes
kill_all_related_processes() {
    # Find all Python processes that might be related to our app
    pids=$(ps aux | grep -E "python.*app\.py|/flask" | grep -v grep | awk '{print $2}')
    if [ -n "$pids" ]; then
        echo "Found the following processes that might be related to our app:"
        for pid in $pids; do
            echo "  PID $pid: $(ps -p $pid -o command= 2>/dev/null || echo 'unknown')"
        done
        
        echo "Sending SIGTERM to all related processes..."
        for pid in $pids; do
            kill -TERM $pid 2>/dev/null
        done
        
        # Wait for processes to exit (max 3 seconds)
        max_wait=3
        for i in $(seq 1 $max_wait); do
            still_running=0
            for pid in $pids; do
                if ps -p $pid > /dev/null 2>&1; then
                    still_running=1
                    break
                fi
            done
            
            if [ $still_running -eq 0 ]; then
                echo "All processes have been terminated"
                break
            fi
            
            echo "Waiting for processes to shut down... ($i/$max_wait)"
            sleep 1
        done
        
        # Force kill any remaining processes
        for pid in $pids; do
            if ps -p $pid > /dev/null 2>&1; then
                echo "Process $pid still running, sending SIGKILL"
                kill -9 $pid 2>/dev/null
            fi
        done
    else
        echo "No related processes found"
    fi
}

# Main script
if [ -f server.pid ]; then
    pid=$(cat server.pid)
    echo "Found server PID file with PID $pid"
    
    # Check if specified process exists
    if ps -p $pid > /dev/null 2>&1; then
        # Send SIGTERM to allow for graceful shutdown (camera release)
        echo "Sending SIGTERM to PID $pid"
        kill -TERM $pid 2>/dev/null
        
        # Wait for process to exit (max 5 seconds)
        max_wait=5
        for i in $(seq 1 $max_wait); do
            if ! ps -p $pid > /dev/null 2>&1; then
                echo "Process $pid has terminated"
                break
            fi
            echo "Waiting for server to shut down gracefully... ($i/$max_wait)"
            sleep 1
        done
        
        # Force kill if still running
        if ps -p $pid > /dev/null 2>&1; then
            echo "Server not responding to SIGTERM, sending SIGKILL"
            kill -9 $pid 2>/dev/null
        fi
    else
        echo "Process with PID $pid not found"
    fi
    
    # Remove PID file
    rm server.pid
else
    echo "No server PID file found"
fi

# Kill any other related processes
kill_all_related_processes

echo "Server stopped"