import boto3
import json
import logging
import os
from typing import Optional, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize AWS clients
bedrock_runtime = boto3.client('bedrock-runtime')
s3_client = boto3.client('s3')

# Environment variables
BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
MODEL_ID = "amazon.nova-reel-v1:1"

class VideoGenerationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, description="Text prompt for video generation")
    seed: Optional[int] = Field(default=42, ge=0, le=2147483646, description="Seed for video generation")

class VideoGenerationResponse(BaseModel):
    job_id: str
    status: str
    message: str
    video_url: Optional[str] = None

async def check_video_generation_status(invocation_arn: str) -> Dict:
    """Check the status of a video generation job"""
    try:
        response = bedrock_runtime.get_async_invoke(
            invocationArn=invocation_arn
        )
        return response
    except Exception as e:
        logger.error(f"Error checking video generation status: {str(e)}")
        raise Exception("Failed to check video generation status")

async def generate_video(request: VideoGenerationRequest) -> Dict:
    """Generate video using Amazon Nova"""
    try:
        # Prepare the request payload according to Nova API specs
        model_input = {
            "taskType": "TEXT_VIDEO",
            "textToVideoParams": {
                "text": request.text
            },
            "videoGenerationConfig": {
                "durationSeconds": 6,
                "fps": 24,
                "dimension": "1280x720",
                "seed": request.seed
            }
        }

        # Configure output location
        output_config = {
            "s3OutputDataConfig": {
                "s3Uri": f"s3://{BUCKET_NAME}"
            }
        }

        # Log the request for debugging
        logger.info(f"Model input: {json.dumps(model_input, indent=2)}")
        logger.info(f"Output config: {json.dumps(output_config, indent=2)}")

        # Start the video generation job
        response = bedrock_runtime.start_async_invoke(
            modelId=MODEL_ID,
            modelInput=model_input,
            outputDataConfig=output_config
        )

        return {
            "invocationArn": response["invocationArn"]
        }

    except Exception as e:
        logger.error(f"Error in video generation: {str(e)}")
        logger.error(f"Full error details: {str(e.__dict__)}")
        raise Exception("Failed to generate video")

def get_video_url(invocation_arn: str) -> str:
    """Generate presigned URL for the video"""
    try:
        job_id = invocation_arn.split("/")[-1]
        return s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': BUCKET_NAME,
                'Key': f'{job_id}/output.mp4'
            },
            ExpiresIn=3600  # URL expires in 1 hour
        )
    except Exception as e:
        logger.error(f"Error generating video URL: {str(e)}")
        raise Exception("Failed to generate video URL")
