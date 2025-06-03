import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_UPLOAD_DIR_SUMM = os.path.join(BASE_DIR, "summarizer", "tmp_uploads")
os.makedirs(TEMP_UPLOAD_DIR_SUMM, exist_ok=True)

PDF_TTL_SECONDS = 3600  # 1 hour TTL for temporary PDFs
CLEANUP_INTERVAL_SECONDS = 1800  # Run cleanup every 30 minutes
