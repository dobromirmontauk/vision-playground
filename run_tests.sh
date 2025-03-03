#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Define color codes
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Run the unit tests
echo -e "${YELLOW}Running unit tests...${NC}"
python -m pytest tests/test_clip_detector.py tests/test_yolo_detector.py tests/test_frame_processor.py tests/test_model_factory.py tests/test_api_endpoints.py -v

# Run integration tests if requested
if [ "$1" == "--with-integration" ]; then
    echo -e "${YELLOW}Running integration tests...${NC}"
    python -m pytest tests/test_integration.py -v
    
    echo -e "${YELLOW}Running frontend tests...${NC}"
    python -m pytest tests/test_frontend.py -v
else
    echo -e "${YELLOW}Skipping integration and frontend tests. Use --with-integration to run them.${NC}"
fi

# Report success
echo -e "${GREEN}All tests completed.${NC}"