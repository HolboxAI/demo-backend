import boto3
import json
import logging
import os
import base64
import uuid
from pathlib import Path
from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create images directory if it doesn't exist
IMAGES_DIR = Path("generated_images")
IMAGES_DIR.mkdir(exist_ok=True)

# Initialize AWS client
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

# Constants
MODEL_ID = "amazon.titan-image-generator-v2:0"

class ImageGenerationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, description="Text prompt for image generation")
    width: int = Field(default=1024, ge=512, le=1024, description="Width of the generated image")
    height: int = Field(default=1024, ge=512, le=1024, description="Height of the generated image")
    number_of_images: int = Field(default=1, ge=1, le=4, description="Number of images to generate")
    cfg_scale: int = Field(default=8, ge=1, le=20, description="CFG scale for image generation")
    seed: Optional[int] = Field(default=42, ge=0, le=2147483646, description="Seed for image generation")
    quality: str = Field(default="standard", description="Quality of the generated image")

class ImageGenerationResponse(BaseModel):
    message: str
    images: List[str]
    local_path: str

async def generate_image(request: ImageGenerationRequest) -> Dict:
    """Generate image using Amazon Titan Image Generator v2"""
    try:
        BACKCEND_URL = os.getenv("BACKEND_URL", "https://demo.holbox.ai/api/demo_backend_v2");
        # Prepare the request payload
        model_input = {
            "textToImageParams": {
                "text": request.text
            },
            "taskType": "TEXT_IMAGE",
            "imageGenerationConfig": {
                "cfgScale": request.cfg_scale,
                "seed": request.seed,
                "quality": request.quality,
                "width": request.width,
                "height": request.height,
                "numberOfImages": request.number_of_images,
            }
        }

        # Log the request for debugging
        logger.info(f"Model input: {json.dumps(model_input, indent=2)}")

        # Generate the image
        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(model_input)
        )

        # Parse the response
        response_body = json.loads(response.get('body').read())
        logger.info(f"Model response: {json.dumps(response_body, indent=2)}")

        if 'images' not in response_body:
            raise HTTPException(status_code=500, detail="No images in response")

        # Process and save images
        image_urls = []
        for image_data in response_body['images']:
            try:
                # Decode and save the image
                if isinstance(image_data, str):
                    image_bytes = base64.b64decode(image_data)
                else:
                    image_bytes = base64.b64decode(image_data.get('data', ''))

                # Generate unique filename and save
                filename = f"{uuid.uuid4()}.png"
                filepath = IMAGES_DIR / filename
                
                with open(filepath, "wb") as f:
                    f.write(image_bytes)
                
                image_urls.append(f"{BACKCEND_URL}/images/{filename}")
                logger.info(f"Image saved successfully: {filename}")

            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to process generated image")

        return {
            "message": "Images generated successfully",
            "images": image_urls,
            "local_path": str(IMAGES_DIR)
        }

    except Exception as e:
        logger.error(f"Error in image generation: {str(e)}")
        logger.error(f"Full error details: {str(e.__dict__)}")
        raise HTTPException(status_code=500, detail=str(e))
    