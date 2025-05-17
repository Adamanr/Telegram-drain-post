import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Telegram API credentials
TELEGRAM_API_ID = os.getenv('TELEGRAM_API_ID')
TELEGRAM_API_HASH = os.getenv('TELEGRAM_API_HASH')
TELEGRAM_CHANNEL_USERNAME = os.getenv('TELEGRAM_CHANNEL_USERNAME', 'behind_the_circus')

# Hugging Face credentials
HF_TOKEN = os.getenv('HF_TOKEN')

# Model configuration
MODEL_NAME = os.getenv('MODEL_NAME', 'mistralai/Mistral-7B-Instruct-v0.1')

# Directory configuration
BASE_DIR = Path(__file__).resolve().parent
POSTS_DIR = os.getenv('POSTS_DIR', 'posts')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'processed_posts')
OUTPUT_IMAGE_DIR = os.getenv('OUTPUT_IMAGE_DIR', '../blog/static/images/posts')
SESSION_NAME = os.getenv('SESSION_NAME', 'channel_to_md')
REQUEST_FILE = os.getenv('REQUEST_FILE', 'request.txt')

# Hugo content directory
HUGO_CONTENT_DIR = os.getenv('HUGO_CONTENT_DIR', '../blog/content/russia/post')

# Worker configuration
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '2'))

def validate_config() -> bool:
    """
    Validate that all required configuration values are present.
    Returns True if all required values are present, otherwise False.
    """
    required_vars = [
        ('TELEGRAM_API_ID', TELEGRAM_API_ID),
        ('TELEGRAM_API_HASH', TELEGRAM_API_HASH),
        ('HF_TOKEN', HF_TOKEN),
    ]
    
    missing_vars = [var_name for var_name, var_value in required_vars if not var_value]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables in your .env file or environment.")
        return False
    
    return True

# Create necessary directories
def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        POSTS_DIR, 
        OUTPUT_DIR, 
        OUTPUT_IMAGE_DIR
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Directory ensured: {dir_path}")

# Validate environment variables when imported
if not validate_config():
    logger.warning("Configuration validation failed. Some features may not work correctly.")

