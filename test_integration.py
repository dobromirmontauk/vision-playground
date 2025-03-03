#!/usr/bin/env python3
"""
Integration tests for YOLO Object Detection application using unittest framework.
Tests all major endpoints and checks server logs for errors.
"""

import unittest
import time
import os
import sys
import json
import urllib.request
import subprocess
import re
import argparse
from datetime import datetime

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def log(message, color=None):
    """Print a colored message to stdout"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if color:
        print(f"{color}[{timestamp}] {message}{RESET}")
    else:
        print(f"[{timestamp}] {message}")

class IntegrationTests(unittest.TestCase):
    """Integration tests for YOLO Object Detection application"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class - parse arguments and check if server is running"""
        parser = argparse.ArgumentParser(description='Run integration tests for YOLO Object Detection app')
        parser.add_argument('--url', default='http://localhost:5000', 
                           help='Base URL of the application server')
        parser.add_argument('--start-server', action='store_true',
                           help='Start a new server instance for testing')
        parser.add_argument('--stop-server', action='store_true',
                           help='Stop the server after testing')
        parser.add_argument('--log-file', default='server.log',
                           help='Path to server log file')
        
        args, _ = parser.parse_known_args()
        
        cls.base_url = args.url
        cls.log_file = args.log_file
        cls.start_server_flag = args.start_server
        cls.stop_server_flag = args.stop_server
        
        log(f"Using base URL: {cls.base_url}", YELLOW)
        log(f"Using log file: {cls.log_file}", YELLOW)
        
        # Start server if requested
        if cls.start_server_flag:
            cls.start_server()
        
        # Check if server is running
        cls.server_available = cls.is_server_running()
        if not cls.server_available:
            log("❗ Server is not available! Some tests may be skipped.", RED)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        if cls.stop_server_flag:
            cls.stop_server()
    
    @classmethod
    def start_server(cls):
        """Start the server using the run_server.sh script"""
        log("Note: Server management should be handled externally. Using --start-server is discouraged.", YELLOW)
        log("Starting server...", YELLOW)
        try:
            subprocess.run(["./run_server.sh"], check=True)
            # Wait for server to start
            time.sleep(5)
            return True
        except subprocess.CalledProcessError as e:
            log(f"❌ Failed to start server: {e}", RED)
            return False
    
    @classmethod
    def stop_server(cls):
        """Stop the server using the stop_server.sh script"""
        log("Note: Server management should be handled externally. Using --stop-server is discouraged.", YELLOW)
        log("Stopping server...", YELLOW)
        try:
            subprocess.run(["./stop_server.sh"], check=True)
            return True
        except subprocess.CalledProcessError as e:
            log(f"❌ Failed to stop server: {e}", RED)
            return False
    
    @classmethod
    def is_server_running(cls):
        """Check if the server is running by making a request"""
        try:
            with urllib.request.urlopen(f"{cls.base_url}/api/stats", timeout=5) as response:
                return response.getcode() == 200
        except Exception:
            return False
    
    def make_request(self, path, method="GET", data=None, headers=None, timeout=10):
        """Make an HTTP request and return response code, body and any exception"""
        url = f"{self.base_url}{path}"
        log(f"Testing endpoint: {method} {url}", YELLOW)
        
        try:
            if data and not isinstance(data, bytes):
                data = json.dumps(data).encode()
            
            if headers is None:
                headers = {}
                
            if data:
                headers['Content-Type'] = 'application/json'
            
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.getcode(), response.read().decode('utf-8'), None
        except Exception as e:
            return None, None, e
    
    def assertEndpointWorks(self, path, method="GET", data=None, headers=None, expected_code=200):
        """Assert that an endpoint returns the expected status code"""
        if not self.server_available:
            self.skipTest("Server is not available")
            
        status, response, error = self.make_request(path, method, data, headers)
        
        self.assertIsNone(error, f"Error accessing {method} {path}: {error}")
        self.assertEqual(status, expected_code, 
                         f"Expected status code {expected_code} but got {status}. Response: {response}")
        return response
    
    def check_server_log_for_errors(self):
        """Check the server log for any errors"""
        if not os.path.exists(self.log_file):
            log(f"❌ Log file {self.log_file} does not exist", RED)
            return False
        
        error_patterns = [
            r"Error",
            r"Exception",
            r"Traceback",
            r"Failed",
            r"Assertion failed",
            r"\[31m",  # Red color in terminal
            r"RuntimeError",
            r"ValueError",
            r"TypeError"
        ]
        
        error_regex = re.compile('|'.join(error_patterns), re.IGNORECASE)
        
        errors_found = []
        with open(self.log_file, 'r') as f:
            for line in f:
                if error_regex.search(line):
                    # Skip known warning about urllib3
                    if "NotOpenSSLWarning" in line:
                        continue
                    # Skip Flask development server warning
                    if "WARNING: This is a development server" in line:
                        continue
                    # Skip "waiting for camera" messages which aren't errors
                    if "No frame available for processing, waiting" in line:
                        continue
                    
                    errors_found.append(line.strip())
        
        return errors_found
    
    def test_01_home_page(self):
        """Test that the home page loads correctly"""
        self.assertEndpointWorks("/")
    
    def test_02_video_feed(self):
        """Test that the video feed endpoint works"""
        if not self.server_available:
            self.skipTest("Server is not available")
            
        url = f"{self.base_url}/video_feed"
        log(f"Testing streaming endpoint: GET {url}", YELLOW)
        
        try:
            # Create request but don't read the full response - just check headers
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as response:
                # Just read a small amount of data to confirm stream starts
                # For MJPEG streams, reading a few bytes is enough to verify the connection
                chunk = response.read(1024)
                
                # Check status and content type
                self.assertEqual(response.getcode(), 200, f"Expected status code 200 but got {response.getcode()}")
                content_type = response.getheader('Content-Type', '')
                self.assertTrue(
                    content_type.startswith('multipart/x-mixed-replace') or 
                    'image' in content_type,
                    f"Expected streaming content type but got: {content_type}"
                )
                self.assertTrue(len(chunk) > 0, "No data received from video stream")
                
                log(f"✅ Video feed endpoint verified - received {len(chunk)} bytes", GREEN)
        except Exception as e:
            self.fail(f"Error accessing video feed: {e}")
    
    def test_03_saved_detections_page(self):
        """Test that the saved detections page loads"""
        self.assertEndpointWorks("/saved_detections")
        
    def test_04_stats_api(self):
        """Test that the stats API works"""
        response = self.assertEndpointWorks("/api/stats")
        
        # Check that the response is valid JSON and has the expected fields
        stats = json.loads(response)
        self.assertIn('fps', stats)
        self.assertIn('median_time', stats)
        self.assertIn('max_time', stats)
    
    def test_05_fix_detection_api(self):
        """Test that the fix detection API works"""
        # Even if there are no actual detections, this should return a valid response
        response = self.assertEndpointWorks("/api/fix_detection", method="POST", data={"id": 0})
        
        # Check that the response is valid JSON
        result = json.loads(response)
        self.assertIn('success', result)
    
    def test_06_check_logs_for_errors(self):
        """Test that the server logs don't contain errors"""
        if not self.server_available:
            self.skipTest("Server is not available")
            
        errors = self.check_server_log_for_errors()
        
        if errors:
            log(f"❌ Found {len(errors)} errors in the log:", RED)
            for i, error in enumerate(errors[:5]):  # Show only first 5 errors
                log(f"  {i+1}. {error}", RED)
            if len(errors) > 5:
                log(f"  ... and {len(errors) - 5} more", RED)
        
        self.assertEqual(len(errors), 0, f"Found {len(errors)} errors in the server log")

if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])