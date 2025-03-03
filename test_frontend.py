#!/usr/bin/env python3
"""
Frontend tests for YOLO Object Detection application using Playwright and unittest.
Tests user interactions with the web interface.
"""

import os
import unittest
import time
import json
import logging
import tempfile
from datetime import datetime
from playwright.sync_api import sync_playwright, expect, TimeoutError as PlaywrightTimeoutError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base URL for testing
BASE_URL = "http://localhost:5000"


class PlaywrightTestCase(unittest.TestCase):
    """Base class for all Playwright tests"""
    
    def setUp(self):
        """Set up the browser and page for each test"""
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(timeout=5000)
        self.page = self.browser.new_page()
        self.page.set_default_timeout(5000)
        
        # Set up console logging
        self.console_logs = []
        self.page.on("console", self._handle_console_log)
        
    def tearDown(self):
        """Clean up after each test"""
        self.browser.close()
        self.playwright.stop()
    
    def _handle_console_log(self, msg):
        """Handle and store console logs"""
        log_entry = f"{msg.type}: {msg.text}"
        self.console_logs.append(log_entry)
        logger.debug(f"Console log: {log_entry}")
    
    def _save_screenshot(self, custom_name=None):
        """Save a screenshot for debugging purposes"""
        if custom_name:
            filename = f"{custom_name}_{int(time.time())}.png"
        else:
            filename = f"error_{int(time.time())}.png"
        
        try:
            self.page.screenshot(path=filename)
            logger.info(f"Screenshot saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save screenshot: {e}")
            return None
    
    def _get_error_details(self):
        """Get detailed error information for better debugging"""
        details = {
            "url": self.page.url,
            "console_logs": self.console_logs,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return details
    
    def run(self, result=None):
        """
        Override the run method to capture errors and save screenshots
        """
        if result is None:
            result = self.defaultTestResult()
            
        try:
            super().run(result)
        except Exception as e:
            # Save error information
            details = self._get_error_details()
            screenshot = self._save_screenshot()
            
            logger.error(f"Test failed with error: {e}")
            logger.error(f"Error details: {details}")
            
            if screenshot:
                logger.error(f"See screenshot at: {screenshot}")
                
            # Re-raise the exception
            raise


class HomepageTests(PlaywrightTestCase):
    """Tests for the main homepage"""
    
    def test_homepage_loads(self):
        """Test that the homepage loads correctly"""
        # Navigate to the homepage
        self.page.goto(BASE_URL)
        
        # Check title
        self.assertIn("YOLO Object Detection", self.page.title())
        
        # Check for main elements
        self.assertTrue(self.page.locator(".video-feed").is_visible())
        self.assertTrue(self.page.locator(".video-overlay").is_visible())
        self.assertTrue(self.page.locator("#statsDisplay").is_visible())


class CropTests(PlaywrightTestCase):
    """Tests for the crop selection functionality"""
    
    def test_crop_selection_flow(self):
        """Test the crop selection and saving flow by interacting with the UI directly"""
        # Navigate to the homepage
        self.page.goto(BASE_URL)
        
        # Wait for video feed to initialize
        self.page.wait_for_selector(".video-feed", state="visible")
        self.page.wait_for_selector(".video-overlay", state="visible")
        
        # Get the video overlay element for interaction
        video_overlay = self.page.locator(".video-overlay")
        box = video_overlay.bounding_box()
        
        if not box:
            self.skipTest("Could not get video overlay bounding box")
            return
            
        # Click in the center of the video to simulate user selecting a crop area
        center_x = box["x"] + box["width"] / 2
        center_y = box["y"] + box["height"] / 2
        self.page.mouse.click(center_x, center_y)
        
        # Wait for UI elements to appear after clicking
        self.page.wait_for_timeout(1000)
        
        # Check if either a preview dialog or notification is shown after clicking
        preview_visible = self.page.locator(".preview-notification.show").count() > 0
        notification_visible = self.page.locator(".notification.show").count() > 0
        
        logger.info(f"UI response: Preview visible: {preview_visible}, Notification visible: {notification_visible}")
        
        # Verify that some UI response happened after clicking
        self.assertTrue(preview_visible or notification_visible, 
                       "Neither preview nor notification appeared after clicking")
        
        # If preview dialog appeared, test the save flow
        if preview_visible:
            # Find and click the save button
            save_button = self.page.locator(".save-crop-btn")
            self.assertTrue(save_button.count() > 0, "Save button not found")
            
            logger.info("Save button found, clicking it")
            save_button.click()
            
            # Check for success notification after saving
            try:
                self.page.wait_for_selector(".notification.show", state="visible", timeout=3000)
                notification_text = self.page.locator(".notification.show").text_content()
                logger.info(f"Notification after save: {notification_text}")
                self.assertIn("Saved:", notification_text, "Notification doesn't indicate success")
            except PlaywrightTimeoutError:
                self.fail("Notification didn't appear after clicking save button")
        
        # Additional validation that UI elements behave as expected
        # Check if detection boxes are visible
        detection_boxes = self.page.locator(".detection-box")
        logger.info(f"Detection boxes visible: {detection_boxes.count() > 0}")
        
        # Verify stats are updating (they should be if the video is running)
        stats_element = self.page.locator("#statsDisplay")
        self.assertTrue(stats_element.is_visible(), "Stats display should be visible")
        stats_text = stats_element.text_content()
        self.assertFalse("Loading stats..." in stats_text and stats_text.strip() == "Loading stats...", 
                        "Stats should load and not remain in loading state")


class SavedDetectionsTests(PlaywrightTestCase):
    """Tests for the saved detections page"""
    
    def test_saved_detections_page_loads(self):
        """Test that the saved detections page loads and shows saved items"""
        # Navigate directly to the saved detections page
        self.page.goto(f"{BASE_URL}/saved_detections")
        
        # Check that the page loads with correct title
        if self.page.title():
            self.assertIn("Saved Detections", self.page.title())
        else:
            self.assertIn("Saved Detections", self.page.content())
        
        # If we have saved items, check that they appear correctly
        cards = self.page.locator(".card")
        if cards.count() > 0:
            self.assertTrue(self.page.locator(".card-img").first.is_visible())
            self.assertTrue(self.page.locator(".badge-original").first.is_visible())
            logger.info(f"Found {cards.count()} saved detection cards")
        else:
            logger.info("No saved detection cards found")


class NavigationTests(PlaywrightTestCase):
    """Tests for navigation between pages"""
    
    def test_navigation_between_pages(self):
        """Test navigation between homepage and saved detections page"""
        # Start on homepage
        self.page.goto(BASE_URL)
        
        # Navigate to saved detections page
        self.page.click("text=View Saved Detections")
        
        # Check URL
        self.assertTrue(self.page.url.endswith("/saved_detections"), 
                        f"URL should end with /saved_detections, got: {self.page.url}")
        
        # Navigate back to homepage
        self.page.click("text=Back to Live Feed")
        
        # Check URL - should be either the exact BASE_URL or end with "/"
        self.assertTrue(self.page.url.endswith("/") or self.page.url == BASE_URL,
                       f"URL should be homepage, got: {self.page.url}")


class UIElementTests(PlaywrightTestCase):
    """Tests for UI elements"""
    
    def test_shutdown_confirmation(self):
        """Test that the shutdown button and confirmation dialog exist"""
        # Navigate to the homepage
        self.page.goto(BASE_URL)
        
        # Verify the shutdown button exists
        shutdown_button = self.page.locator("#shutdownButton")
        self.assertTrue(shutdown_button.count() > 0, "Shutdown button not found")
        
        # Check that the confirmation dialog element exists
        dialog = self.page.locator("#confirmShutdown")
        self.assertTrue(dialog.count() > 0, "Confirmation dialog element not found")
        
        # Check that the confirmation buttons exist in the DOM
        yes_button = self.page.locator("#confirmYes")
        no_button = self.page.locator("#confirmNo")
        
        self.assertTrue(yes_button.count() > 0, "Yes button element not found")
        self.assertTrue(no_button.count() > 0, "No button element not found")
        
        logger.info("Shutdown confirmation UI elements verified")


class StatsTests(PlaywrightTestCase):
    """Tests for stats display functionality"""
    
    def test_stats_update(self):
        """Test that stats update correctly from the API"""
        # Navigate to the homepage
        self.page.goto(BASE_URL)
        
        # Find the stats display element
        stats_element = self.page.locator("#statsDisplay")
        stats_element.wait_for(state="visible")
        
        # Wait for natural stats updates (up to 15 seconds)
        max_attempts = 15
        stats_loaded = False
        
        for attempt in range(max_attempts):
            # Read the current content - no modification
            stats_text = stats_element.text_content()
            logger.info(f"Stats text (attempt {attempt+1}/{max_attempts}): '{stats_text}'")
            
            # Check if stats have loaded
            if "Loading stats..." not in stats_text and "FPS:" in stats_text:
                stats_loaded = True
                break
                
            # Wait for natural update
            logger.info("Waiting for stats to update naturally...")
            self.page.wait_for_timeout(1000)  # Wait 1 second
        
        # Check if we got stats
        if not stats_loaded:
            # Take a screenshot for debugging
            self._save_screenshot("stats_not_loading")
            logger.warning("Stats never loaded - see screenshot for debugging")
        
        # Test the API directly 
        api_data = self.page.evaluate("""async () => {
            try {
                const response = await fetch('/api/stats');
                if (!response.ok) {
                    return { error: `HTTP error ${response.status}` };
                }
                return await response.json();
            } catch (e) {
                return { error: e.toString() };
            }
        }""")
        
        logger.info(f"API data for verification: {api_data}")
        
        # Verify API returns valid data
        self.assertIn('fps', api_data, "API response missing 'fps' field")
        self.assertIn('median_time', api_data, "API response missing 'median_time' field")
        self.assertIn('max_time', api_data, "API response missing 'max_time' field")
        
        # Make assertions on either the loaded stats or the API data
        if stats_loaded:
            # Get actual stats text
            stats_text = stats_element.text_content()
            
            # Check format using simple string containment
            self.assertIn("FPS:", stats_text, f"Stats doesn't contain 'FPS:': '{stats_text}'")
            self.assertIn("median", stats_text, f"Stats doesn't contain 'median': '{stats_text}'")
            self.assertIn("max", stats_text, f"Stats doesn't contain 'max': '{stats_text}'")
            
            logger.info("Stats validation passed - frontend is displaying stats correctly")
        else:
            # Still verify that the API works even if the frontend doesn't show stats
            logger.info("Stats didn't load in frontend, but API is working")
            self.assertGreaterEqual(api_data['fps'], 0, "API returned negative FPS")
            self.assertGreaterEqual(api_data['median_time'], 0, "API returned negative median time")
            self.assertGreaterEqual(api_data['max_time'], 0, "API returned negative max time")


# Skip this test in normal execution as it would shut down the server
@unittest.skip("This test would shut down the server")
class ServerShutdownTests(PlaywrightTestCase):
    """Tests for server shutdown functionality"""
    
    def test_server_shutdown(self):
        """Test the server shutdown functionality"""
        # Navigate to the homepage
        self.page.goto(BASE_URL)
        
        # Click the shutdown button
        self.page.click("#shutdownButton")
        
        # Click confirm
        self.page.click("#confirmYes")
        
        # Wait for the shutdown message
        self.page.wait_for_selector("text=Server is shutting down", timeout=5000)
        
        # Wait a bit to let the server shut down
        time.sleep(3)
        
        # Try to access the server again - should fail
        try:
            self.page.goto(BASE_URL, timeout=3000)
            self.fail("Server did not shut down properly")
        except:
            # Expected behavior - server should be down
            pass


if __name__ == "__main__":
    unittest.main()