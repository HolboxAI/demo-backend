import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_UPLOAD_DIR = os.path.join(BASE_DIR, "tmp_uploads")
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

PDF_TTL_SECONDS = 3600  # 1 hour TTL for demo
CLEANUP_INTERVAL_SECONDS = 1800  # Cleanup every 30 mins
