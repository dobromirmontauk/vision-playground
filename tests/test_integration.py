"""
Integration tests for the vision-playground application.

These tests require a running server instance.
"""

import os
import time
import pytest
import requests
import json
import subprocess
import signal
from pathlib import Path


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the application."""
    
    server_url = "http://localhost:5000"
    server_process = None
    
    @pytest.fixture(scope="class", autouse=True)
    def start_server(self, request):
        """Start the server before running the tests and stop it after."""
        # Don't try to start a new server if one is already running
        response = None
        try:
            response = requests.get(f"{self.server_url}/api/stats", timeout=0.5)
        except requests.exceptions.ConnectionError:
            pass
            
        if response is not None and response.status_code == 200:
            # Server is already running, don't start a new one
            print("Using existing server for tests.")
            yield
            return
            
        # Start a new server process
        print("Starting server for integration tests...")
        startup_script = Path(os.path.dirname(os.path.dirname(__file__))) / "run_server.sh"
        self.server_process = subprocess.Popen(["bash", str(startup_script)])
        
        # Wait for server to start
        max_retries = 10
        for i in range(max_retries):
            try:
                response = requests.get(f"{self.server_url}/api/stats")
                if response.status_code == 200:
                    print(f"Server started successfully after {i+1} attempts.")
                    break
            except requests.exceptions.ConnectionError:
                pass
                
            if i < max_retries - 1:
                time.sleep(2)
                
        # Register finalizer to stop server
        def fin():
            if self.server_process:
                print("Stopping server...")
                self.server_process.send_signal(signal.SIGTERM)
                self.server_process.wait(timeout=10)
                print("Server stopped.")
                
        request.addfinalizer(fin)
        
        yield
    
    def test_index_page(self):
        """Test that the index page loads."""
        response = requests.get(self.server_url)
        assert response.status_code == 200
        assert "text/html" in response.headers["Content-Type"]
    
    def test_stats_endpoint(self):
        """Test that the stats endpoint returns data."""
        response = requests.get(f"{self.server_url}/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "fps" in data
        assert "median_time" in data
        assert "max_time" in data
    
    def test_frame_data_endpoint(self):
        """Test that the frame data endpoint returns data."""
        response = requests.get(f"{self.server_url}/api/frame_data")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "frame" in data
        assert "dimensions" in data
        assert "detections" in data


def test_server_connection():
    """Test that the server is reachable."""
    try:
        response = requests.get("http://localhost:5000/api/stats", timeout=0.5)
        assert response.status_code == 200
        print("Server is running and reachable.")
    except requests.exceptions.ConnectionError:
        pytest.skip("Server is not running. Skipping integration tests.")