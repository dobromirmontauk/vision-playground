"""
Frontend tests for the vision-playground application.

These tests use Playwright to test the frontend UI.
"""

import os
import time
import pytest
import requests
import subprocess
import signal
from pathlib import Path
from playwright.sync_api import Page, expect


@pytest.mark.integration
class TestFrontend:
    """Test suite for the frontend UI."""
    
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
        print("Starting server for frontend tests...")
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
    
    def test_main_page_loads(self, page: Page):
        """Test that the main page loads correctly."""
        # Go to the application
        page.goto(self.server_url)
        
        # Check that the page title contains "Object Detection"
        expect(page).to_have_title(lambda t: "Detection" in t)
        
        # Check that the video container is present
        video_container = page.locator(".video-container")
        expect(video_container).to_be_visible()
        
        # Check that the video feed is present
        video_feed = page.locator(".video-feed")
        expect(video_feed).to_be_visible()
        
        # Check that the stats are displayed
        # Wait for stats to be loaded (they're loaded asynchronously)
        page.wait_for_selector(".stats-container")
        stats_display = page.locator(".stats")
        expect(stats_display).to_be_visible()
        
        # Allow time for stats to update
        page.wait_for_timeout(2000)
        stats_text = stats_display.text_content()
        assert "FPS" in stats_text
    
    def test_saved_detections_page(self, page: Page):
        """Test that the saved detections page loads correctly."""
        # Go to the saved detections page
        page.goto(f"{self.server_url}/saved_detections")
        
        # Check that the page loads
        expect(page).to_have_title(lambda t: "Saved Detections" in t)
        
        # Check for the presence of key elements (note: might be empty if no detections saved)
        page.wait_for_selector("h1")
        header = page.locator("h1")
        expect(header).to_contain_text("Saved Detections")


# Skip all tests if no Playwright browser is installed
def test_playwright_available():
    """Check if Playwright is properly installed."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            browser.close()
    except Exception as e:
        pytest.skip(f"Playwright not available: {str(e)}")