# virtual_try_on/virtual_tryon.py

import aiohttp
import uuid
import asyncio
import os
import base64
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("FASHN_API_URL")
AUTH_KEY = os.getenv("FASHN_AUTH_KEY")

if not API_URL or not AUTH_KEY:
    raise ValueError("Missing FASHN_API_URL or FASHN_AUTH_KEY in environment")

class VirtualTryOnRequest(BaseModel):
    model_image: str  # Base64 encoded image
    garment_image: str  # Base64 encoded image
    category: str = "tops"

class VirtualTryOnResponse(BaseModel):
    id: str

class StatusResponse(BaseModel):
    status: str
    output: Optional[list] = None

# Global storage for processing jobs (equivalent to React state)
processing_jobs = {}

async def handle_process(request: VirtualTryOnRequest) -> VirtualTryOnResponse:
    """
    Equivalent to handleProcess function in React
    """
    # Debug: print first few characters to verify inputs
    print("model_image:", request.model_image[:30], "...", "garment_image:", request.garment_image[:30], "...")

    # Input validation
    if not request.model_image or not request.garment_image:
        print("ERROR: One or both images are missing or empty!")
        raise HTTPException(status_code=400, detail="Both model_image and garment_image are required and must be base64 strings.")
    
    # Optionally: check for valid base64 (optional, for stricter validation)
    try:
        base64.b64decode(request.model_image)
        base64.b64decode(request.garment_image)
    except Exception:
        print("ERROR: One or both images are not valid base64!")
        raise HTTPException(status_code=400, detail="Both images must be valid base64-encoded strings.")
    
    job_id = str(uuid.uuid4())
    processing_jobs[job_id] = {
        "status": "processing",
        "output": None
    }
    
    payload = {
        "model_image": request.model_image,
        "garment_image": request.garment_image, 
        "category": request.category,
    }

    try:
        headers = {
            "Authorization": f"Bearer {AUTH_KEY}", 
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{API_URL}/run", json=payload, headers=headers) as response:
                print(f"POST {API_URL}/run => {response.status}")
                if response.status != 200:
                    error_text = await response.text()
                    print("API Error Text:", error_text)
                    processing_jobs[job_id]["status"] = "failed"
                    raise HTTPException(status_code=400, detail=f"API Error: {error_text}")
                
                result = await response.json()
                print("API response:", result)
                fashn_id = result.get("id")
                if not fashn_id:
                    print("No 'id' returned from API!")
                    processing_jobs[job_id]["status"] = "failed"
                    raise HTTPException(status_code=500, detail="No 'id' returned from Fashn API.")
                
                # Start background polling
                asyncio.create_task(poll_prediction_status(job_id, fashn_id))
                
                return VirtualTryOnResponse(id=job_id)

    except Exception as error:
        print("Exception in handle_process:", error)
        processing_jobs[job_id]["status"] = "failed"
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(error)}")

async def poll_prediction_status(job_id: str, fashn_id: str):
    """
    Equivalent to pollPredictionStatus function in React
    """
    try:
        headers = {
            "Authorization": f"Bearer {AUTH_KEY}", 
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.get(f"{API_URL}/status/{fashn_id}", headers=headers) as response:
                        print(f"GET {API_URL}/status/{fashn_id} => {response.status}")
                        if response.status == 200:
                            resp_result = await response.json()
                            print("Poll result:", resp_result)
                            
                            if resp_result["status"] == "completed":
                                processing_jobs[job_id]["status"] = "completed"
                                processing_jobs[job_id]["output"] = resp_result["output"]
                                break
                                
                            elif resp_result["status"] == "failed":
                                processing_jobs[job_id]["status"] = "failed"
                                print("Image processing failed...")
                                break
                    
                    # Wait before polling again
                    await asyncio.sleep(3)
                    
                except Exception as e:
                    print(f"Polling error: {e}")
                    await asyncio.sleep(3)
                    
    except Exception as e:
        processing_jobs[job_id]["status"] = "failed"
        print(f"Polling exception: {e}")

async def get_status(job_id: str) -> StatusResponse:
    """
    Get status of processing job
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    return StatusResponse(
        status=job["status"],
        output=job.get("output")
    )
