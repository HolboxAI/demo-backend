import os
import time
import asyncio
from .config import TEMP_UPLOAD_DIR, PDF_TTL_SECONDS, CLEANUP_INTERVAL_SECONDS
from .embeddings import delete_vectors_by_pdf_id

async def cleanup_task():
    while True:
        now = time.time()
        deleted_files = []
        for filename in os.listdir(TEMP_UPLOAD_DIR):
            filepath = os.path.join(TEMP_UPLOAD_DIR, filename)
            if not os.path.isfile(filepath):
                continue
            file_age = now - os.path.getmtime(filepath)
            if file_age > PDF_TTL_SECONDS:
                try:
                    os.remove(filepath)
                    delete_vectors_by_pdf_id(filename)
                    deleted_files.append(filename)
                except Exception as e:
                    print(f"Failed to delete {filename}: {e}")
        if deleted_files:
            print(f"Cleanup: Removed PDFs and vectors for {deleted_files}")
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
