import logging
import os
from datetime import datetime

# Create logs directory
LOGS_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Define log file with timestamp
LOG_FILE_PATH = os.path.join(LOGS_DIR, f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log")

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True
)

# Expose logger instance
logger = logging.getLogger(__name__)
