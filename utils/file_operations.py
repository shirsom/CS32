import os
from config import VALID_FILE_EXTENSIONS

def is_valid_image_file(filename):
  # Check file name extension
  if os.path.splitext(filename)[1].lower() not in VALID_FILE_EXTENSIONS:
    print(f"Invalid image file extension \"{filename}\". Skipping this file...")
    return False

  return True
