from requests import status_codes
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request, WebSocket, status, Query, Form, Depends, Body
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import shutil
from pydantic import BaseModel
from werkzeug.utils import secure_filename
from datetime import datetime
import asyncio
from pathlib import Path
from fastapi.staticfiles import StaticFiles
import traceback
import logging
import base64
import json
from auth import get_current_user  # Import the authentication dependency
import httpx
import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from urllib.parse import urlparse


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
#calorie_content imports
from typing import Optional

from calorie_content.calorie_content import analyze_calories_flow

#nl2sql imports
from nl2sql.nl2sql import ask_nl2sql
from nl2sql.Routes.api import router as nl2sql_router

# Virtual try-on imports
from virtual_try_on.virtual_try_on import VirtualTryOnRequest, VirtualTryOnResponse, StatusResponse
from virtual_try_on.virtual_try_on import get_status
from virtual_try_on.virtual_try_on import handle_process

## Healthscribe imports
from healthscribe.healthscribe import allowed_file, upload_to_s3, fetch_summary, start_transcription, ask_claude

# # Face detection imports
from face_detection.face_detection import process_video_frames
# Face recognigation imports
from face_recognigation.face_recognigation import add_face_to_collection, recognize_face,add_face_and_upload
from face_recognigation.face_recognigation import get_rekognition_client_accountB, FACE_COLLECTION_ID
#eda imports
from eda.eda import (
    generate_graph_from_session, 
    ask_openai_from_session, 
    get_csv_info,
    file_manager
) 
# In your app.py (or where you're using the models)
from face_recognigation._component.model import SessionLocal, UserMetadata
from face_recognigation.face_recognigation import delete_face_by_photo
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


# # PDF data extraction imports
# from pdf_data_extraction.app.config import TEMP_UPLOAD_DIR
# from pdf_data_extraction.app.pdf_utils import extract_text_from_pdf, chunk_text
# from pdf_data_extraction.app.embeddings import store_embeddings, query_embeddings, generate_answer
# from pdf_data_extraction.app.models import UploadResponse, QuestionRequestPDF, AnswerResponse
# from pdf_data_extraction.app.cleanup import cleanup_task

from ddx.ddx import DDxAssistant
from pii_redactor.redactor import PiiRedactor
from pii_extractor.extractor import PiiExtractor

 #Text to Image imports
from txt2img.main import ImageGenerationRequest, ImageGenerationResponse, generate_image

#Text to video imports
from txt2vid.main import (
    VideoGenerationRequest,
    VideoGenerationResponse,
    generate_video,
    check_video_generation_status,
    get_video_url
)
## summarizer imports
from summarizer.models import UploadResponse, SummaryRequest, SummaryResponse
from summarizer.cleanup import cleanup_task_summ
from summarizer.config import TEMP_UPLOAD_DIR_SUMM
from summarizer.pdf_utils import extract_text_from_pdf
from summarizer.openai_utils import generate_summary


## Voice agent imports
from voice_agent.voice_agent import voice_websocket_endpoint


# handwritten
from handwritten.handwritten import extract_handwritten_text

# Marketplace
from marketplace.fulfillment import router as marketplace_router

from ai_concierge.ai_concierge import ai_concierge

# Dependency to get the Authorization token
def get_authorization_header(request: Request):
    authorization_header = request.headers.get("Authorization")
    if authorization_header is None:
        raise HTTPException(status_code=401, detail="Authorization header is missing")
    
    # You can also add token validation here if needed
    token = authorization_header.split("Bearer ")[-1]  # Extract the token
    if not token:
        raise HTTPException(status_code=401, detail="Invalid Authorization format")
    
    return token  # Return the token to use in the endpoint functions

api_router = APIRouter(dependencies=[Depends(get_current_user)])


# Initialize FastAPI app
# app = FastAPI(dependencies=[Depends(get_authorization_header)]) # All endpoints will require authentication by default
app = FastAPI() 
app.include_router(api_router)
app.include_router(marketplace_router)
# organization router

# # Initialize instances of your assistants
ddx_assistant = DDxAssistant()
pii_redactor = PiiRedactor()
pii_extractor = PiiExtractor()


class QuestionRequest(BaseModel):
    question: str

class PiiRequest(BaseModel):
    text: str

class AgentCoreRequest(BaseModel):
    prompt: str


# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create images directory if it doesn't exist
IMAGES_DIR = Path("generated_images")
IMAGES_DIR.mkdir(exist_ok=True)
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


load_dotenv()  # Load environment variables from .env file

AGENTCORE_URL = os.getenv(
    "AGENTCORE_URL",
    "https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/arn%3Aaws%3Abedrock-agentcore%3Aus-east-1%3A992382417943%3Aruntime%2Fstrands_agent_file_system-fd9ElvFsMQ/invocations?qualifier=DEFAULT"
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Dependency to get the Authorization token
def get_authorization_header(request: Request):
    authorization_header = request.headers.get("Authorization")
    if authorization_header is None:
        raise HTTPException(status_code=401, detail="Authorization header is missing")
    
    # You can also add token validation here if needed
    token = authorization_header.split("Bearer ")[-1]  # Extract the token
    if not token:
        raise HTTPException(status_code=401, detail="Invalid Authorization format")
    
    return token  # Return the token to use in the endpoint functions


#face_recognigation database setup
# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

#calorie_content database setup
def _ext_from_mime(mime: str) -> str:
    mapping = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/heic": ".heic",
        "image/heif": ".heif",
    }
    return mapping.get((mime or "").lower(), ".bin")       


@app.post("/api/demo_backend_v2/detect_faces")
async def detect_faces_api(video: UploadFile = File(...)):
    """
    API endpoint to upload a video, process it for face recognition, and return detection results.
    """
    if not video.filename:
        raise HTTPException(status_code=400, detail="Invalid file name.")

    # Save uploaded video to a file
    file_path = os.path.join(UPLOAD_FOLDER, video.filename)
    with open(file_path, "wb") as f:
        f.write(await video.read())

    # Call the face detection function from face_detection module
    detected_faces = process_video_frames(file_path)

    # Prepare the response
    response = {
        "video": video.filename,
        "detected_faces": detected_faces
    }

    return JSONResponse(content=response)




# @app.post("/api/demo_backend_v2/add_face")
# async def add_user_face_api(image: UploadFile = File(...), name: str = Form(...), age: int = Form(None), gender: str = Form(None),token: str = Depends(get_authorization_header)):
#     """
#     API endpoint to add a face to the collection and store user data in RDS.
#     """
#     if not image.filename:
#         raise HTTPException(status_code=400, detail="Invalid file name.")

#     # Save uploaded image to a file
#     file_path = os.path.join(UPLOAD_FOLDER, image.filename)
#     with open(file_path, "wb") as f:
#         f.write(await image.read())

#     # Add face to collection and store metadata in RDS
#     result = add_face_to_collection(file_path, name)  # This function adds the face to Rekognition

#     if "face_id" not in result:
#         raise HTTPException(status_code=500, detail="Face addition to Rekognition failed.")

#     # Store user metadata in RDS
#     db_session = SessionLocal()
#     user_metadata = UserMetadata(
#         face_id=result['face_id'],  # The face_id returned from Rekognition
#         name=name,
#         age=age,
#         gender=gender,
#         timestamp=datetime.now()
#     )

#     db_session.add(user_metadata)  # Add the user metadata to the session
#     db_session.commit()  # Commit the transaction to save it to the database
#     db_session.close()  # Close the session

#     # Clean up the temporary file
#     os.remove(file_path)

#     return {"message": "Face added and metadata saved successfully", "face_id": result['face_id']}
@app.post("/api/demo_backend_v2/add_face")
async def add_user_face_api(
    
    image: UploadFile = File(...),
    name: str = Form(...),
    background_tasks:  BackgroundTasks = BackgroundTasks(),
    age: int = Form(None),
    gender: str = Form(None),
    token: str = Depends(get_authorization_header),
    
     # Use BackgroundTasks directly, no Depends
):
    """
    API endpoint to add a face to the collection, upload the image to S3 asynchronously, and store user data in RDS.
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="Invalid file name.")

    # Add face to Rekognition, upload to S3, and store user metadata
    face_id, image_url = await add_face_and_upload(image, name, background_tasks)

    # Store user metadata in RDS
    db_session = SessionLocal()
    user_metadata = UserMetadata(
        face_id=face_id,  # The face_id returned from Rekognition
        name=name,
        age=age,
        gender=gender,
        timestamp=datetime.now(),
        image_url=image_url  # Save the image URL in the user metadata table
    )

    db_session.add(user_metadata)
    db_session.commit()
    db_session.close()

    return {"message": "Face added, image uploaded to S3, and metadata saved successfully", "face_id": face_id, "image_url": image_url}





@app.post("/api/demo_backend_v2/recognize_face")
async def recognize_face_api(image: UploadFile = File(...)):
    """
    API endpoint to recognize a face from the collection and retrieve associated user data.
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="Invalid file name.")

    # Save uploaded image to a file
    file_path = os.path.join(UPLOAD_FOLDER, image.filename)
    with open(file_path, "wb") as f:
        f.write(await image.read())

    # Recognize face using Rekognition
    result = recognize_face(file_path)  # This function searches for the face in Rekognition's collection

    logger.info(f"Recognition result: {result}")  # Log the result for debugging

    if result["recognized"]:
        face_id = result["face_id"]  # Use face_id to query database instead of 'name'
        logger.info(f"Recognized face_id: {face_id}")  # Log the face_id

        # Retrieve user metadata from RDS using face_id
        db_session = SessionLocal()
        user_metadata = db_session.query(UserMetadata).filter(UserMetadata.face_id == face_id).first()

        if user_metadata:
            logger.info(f"User metadata found: {user_metadata.name}, {user_metadata.age}, {user_metadata.gender}")
            
            result.update({
                "user_name": user_metadata.name,
                "user_age": user_metadata.age,
                "user_gender": user_metadata.gender,
                "user_timestamp": user_metadata.timestamp,
                "user_id": user_metadata.id  # Add all metadata fields you need
            })
        else:
            result.update({
                "message": "Face recognized, but no user data found"
            })

        db_session.close()
    else:
        logger.info("No face recognized")

    # Clean up the temporary file
    os.remove(file_path)

    return result





@app.on_event("startup")
async def startup_event():
    # asyncio.create_task(cleanup_task())
    asyncio.create_task(cleanup_task_summ())
    asyncio.create_task(cleanup_eda_sessions())


@app.post("/api/demo_backend_v2/summarizer/upload_pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    pdf_id = str(uuid.uuid4())
    save_path = os.path.join(TEMP_UPLOAD_DIR_SUMM, f"{pdf_id}.pdf")

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return UploadResponse(pdf_id=pdf_id, message="PDF uploaded successfully")

@app.post("/api/demo_backend_v2/summarizer/get_summary", response_model=SummaryResponse)
async def get_summary(request: SummaryRequest):
    pdf_path = os.path.join(TEMP_UPLOAD_DIR_SUMM, f"{request.pdf_id}.pdf")
    if not os.path.isfile(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found or expired")

    text = extract_text_from_pdf(pdf_path)
    # Optional: if text is huge, consider chunking here (not included for simplicity)
    summary = generate_summary(text)

    return SummaryResponse(summary=summary)



@app.post("/api/demo_backend_v2/ddx")
async def ask_ddx(question_request: QuestionRequest, token: str = Depends(get_authorization_header)):
    # logger.info(f"Received Authorization Token: {token}")
    # Now you can use the token for validation or other purposes
    response = ddx_assistant.ask(question_request.question)
    return {"answer": response}


  
@app.post("/api/demo_backend_v2/redact")
async def redact_pii(request: PiiRequest):
    redacted_text = pii_redactor.redact(request.text)
    return {"redacted": redacted_text}


@app.post("/api/demo_backend_v2/extract")
async def extract_pii(request: PiiRequest):
    extracted = pii_extractor.extract(request.text)
    return {"extracted": extracted}


@app.post("/api/demo_backend_v2/healthscribe/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    try:
        if not file.filename or not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file format")

        filename = secure_filename(file.filename)
        local_path = os.path.join(UPLOAD_FOLDER, filename)

        # Save uploaded file locally
        with open(local_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Upload file to S3
        file_url = upload_to_s3(local_path, filename)

        return {"fileUrl": file_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/demo_backend_v2/healthscribe/question-ans")
async def question_answer(req: QuestionRequest):
    global transcription_summary
    if not transcription_summary:
        raise HTTPException(status_code=400, detail="Transcription summary not available. Complete transcription first.")

    question = req.question
    if not question:
        raise HTTPException(status_code=400, detail="No question provided.")

    try:
        answer = ask_claude(question, transcription_summary)
        return {"question": question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@app.post("/api/demo_backend_v2/healthscribe/start-transcription")
async def start_transcription_route(request: Request):
    global transcription_summary
    BUCKET_NAME = os.getenv('BUCKET_NAME')
    data = await request.json()
    audio_url = data.get('audioUrl')
    

    if not audio_url:
        raise HTTPException(status_code=400, detail="Audio URL is required.")
    
    # Check if the audio is from the predefinedAudios folder
    if "predefinedAudios/" in audio_url:
        try:
            # Extract filename (e.g., predefinedAudios1.mp3 → predefinedAudios1)
            filename = os.path.basename(audio_url).rsplit('.', 1)[0]
            # Construct the expected S3 URI for summary.json
            summary_s3_uri = f"s3://{BUCKET_NAME}/health_scribe/predefinedAudios/{filename}_summary.json"
            
            # return {"TranscriptFileUri": summary_s3_uri}
            transcription_summary = fetch_summary(summary_s3_uri)
            return {"summary": transcription_summary}
        except Exception as e:
            raise Exception(f"Error constructing predefined summary URI: {e}")

    try:

        S3_PUBLIC_PREFIX = f"https://{BUCKET_NAME}.s3.amazonaws.com/"
        S3_PRIVATE_PREFIX = f"s3://{BUCKET_NAME}/"

        if audio_url.startswith(S3_PUBLIC_PREFIX):
            audio_url = S3_PRIVATE_PREFIX + audio_url[len(S3_PUBLIC_PREFIX):]
            

        PREDEFINED_PREFIX = f"s3://{BUCKET_NAME}/predefined/"
        if audio_url.startswith(PREDEFINED_PREFIX):
            

            filename = os.path.basename(audio_url)
            summary_filename = f"summary_{filename.replace('.mp3', '.json')}"
            summary_s3_key = f"predefinedAudios/{summary_filename}"

            transcription_summary = fetch_summary(f"s3://{BUCKET_NAME}/{summary_s3_key}")
            return {"summary": transcription_summary}

        # If it's a local file path, upload to S3 first
        if not audio_url.startswith("s3://"):
            if os.path.exists(audio_url):
                filename = os.path.basename(audio_url)
                audio_url = upload_to_s3(audio_url, filename)
            else:
                raise HTTPException(status_code=400, detail="Invalid audio file path.")

        job_name = f"medi_trans_{int(datetime.now().strftime('%Y_%m_%d_%H_%M'))}"
        
        medical_scribe_output = start_transcription(job_name, audio_url)

        if "ClinicalDocumentUri" in medical_scribe_output:
            summary_uri = medical_scribe_output['ClinicalDocumentUri']
            transcription_summary = fetch_summary(summary_uri)
        else:
            transcription_summary = medical_scribe_output.get('ClinicalDocumentText', "No summary found.")

        return {"summary": transcription_summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/api/demo_backend_v2/healthscribe/start-transcription")
# async def start_transcription_route(request: Request):
#     global transcription_summary

#     data = await request.json()
#     audio_url = data.get('audioUrl')
#     print("Received Audio URL:", audio_url)

#     if not audio_url:
#         raise HTTPException(status_code=400, detail="Audio URL is required.")

#     try:
#         BUCKET_NAME = os.getenv('BUCKET_NAME')
#         S3_PUBLIC_PREFIX = f"https://{BUCKET_NAME}.s3.amazonaws.com/"
#         S3_PRIVATE_PREFIX = f"s3://{BUCKET_NAME}/"

#         if audio_url.startswith(S3_PUBLIC_PREFIX):
#             audio_url = S3_PRIVATE_PREFIX + audio_url[len(S3_PUBLIC_PREFIX):]
#             print("Converted to S3 URL:", audio_url)

#         PREDEFINED_PREFIX = f"s3://{BUCKET_NAME}/predefined/"
#         if audio_url.startswith(PREDEFINED_PREFIX):
#             print("Predefined audio detected. Fetching existing summary...")

#             filename = os.path.basename(audio_url)
#             summary_filename = f"summary_{filename.replace('.mp3', '.json')}"
#             summary_s3_key = f"predefined/{summary_filename}"

#             transcription_summary = fetch_summary(f"s3://{BUCKET_NAME}/{summary_s3_key}")
#             return {"summary": transcription_summary}

#         # If it's a local file path, upload to S3 first
#         if not audio_url.startswith("s3://"):
#             if os.path.exists(audio_url):
#                 filename = os.path.basename(audio_url)
#                 audio_url = upload_to_s3(audio_url, filename)
#             else:
#                 raise HTTPException(status_code=400, detail="Invalid audio file path.")

#         job_name = f"medi_trans_{int(datetime.now().strftime('%Y_%m_%d_%H_%M'))}"
#         print("Starting transcription job:", job_name)
#         medical_scribe_output = start_transcription(job_name, audio_url)

#         if "ClinicalDocumentUri" in medical_scribe_output:
#             summary_uri = medical_scribe_output['ClinicalDocumentUri']
#             transcription_summary = fetch_summary(summary_uri)
#         else:
#             transcription_summary = medical_scribe_output.get('ClinicalDocumentText', "No summary found.")

#         return {"summary": transcription_summary}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/demo_backend_v2/nl2sql/ask")
async def ask_nl2sql_endpoint(request: QuestionRequest):
    try:
        response = ask_nl2sql(request.question)
        return {"answer": response}
    except Exception as e:
        logger.error(f"Error in nl2sql endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing NL2SQL request.")



#Virtual try on backend API endpoints
# @app.post("/api/demo_backend_v2/virtual-tryon/run", response_model=VirtualTryOnResponse)
# async def virtual_tryon_run(request: VirtualTryOnRequest):
#     """
#     Process virtual try-on request (equivalent to handleProcess in React)
#     """
#     try:
        
#         result = await handle_process(request)
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Virtual try-on processing failed: {str(e)}")

#Virtual try on backend API endpoints
@app.post("/api/demo_backend_v2/virtual-tryon/run", response_model=VirtualTryOnResponse)
async def virtual_tryon_run(request: VirtualTryOnRequest):
    """
    Process virtual try-on request (equivalent to handleProcess in React)
    """
    try:
        
        result = await handle_process(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Virtual try-on processing failed: {str(e)}")

@app.get("/api/demo_backend_v2/virtual-tryon/status/{job_id}", response_model=StatusResponse)
async def virtual_tryon_status(job_id: str):
    """
    Get the status of a virtual try-on job (equivalent to pollPredictionStatus in React)
    """
    try:
        
        result = await get_status(job_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")


@app.post(
    "/api/demo_backend_v2/generate-video",
    response_model=VideoGenerationResponse
)
async def create_video_generation(
    request: VideoGenerationRequest,
):
    """
    Generate a video from a text prompt using Amazon Nova.
    """
    try:
        result = await generate_video(request)
        # Generate unique job ID
        job_id = result["invocationArn"]
                
        return VideoGenerationResponse(
            job_id=job_id,
            status="processing",
            message="Video generation started successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to start video generation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start video generation: {str(e)}"
        )

@app.get(
    "/api/demo_backend_v2/video-status",
    response_model=VideoGenerationResponse
)
async def get_video_status(job_id: str = Query(..., description="Full invocation ARN")):
    """
    Check the status of a video generation job.
    """
    try:
        status = await check_video_generation_status(job_id)
        
        if status["status"] == "Completed":
            video_url = get_video_url(job_id)
            return VideoGenerationResponse(
                job_id=job_id,
                status="completed",
                message="Video generation completed successfully",
                video_url=video_url
            )
            
        return VideoGenerationResponse(
            job_id=job_id,
            status=status["status"].lower(),
            message=f"Video generation {status['status'].lower()}"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Failed to get video generation status"
        )
    

    

# Voice Agent WebSocket Endpoint
# @app.websocket("/api/demo_backend_v2/voice_agent/voice")
# async def websocket_route(ws: WebSocket):
#     print("WebSocket connection opened")
#     await ws.accept()
#     try:
#         while True:
#             data = await ws.receive_text()
#             print("Received data:", data)
#             await ws.send_text(f"Message received: {data}")
#     except Exception as e:
#         print(f"Error during WebSocket communication: {e}")
#     finally:
#         print("WebSocket connection closed")
#         await ws.close()

@app.websocket("/api/demo_backend_v2/voice_agent/voice")
async def websocket_route(ws: WebSocket):
    await ws.accept()
    print("WebSocket connection established")
    
    try:
        # Wait for the first message containing the token
        message = await ws.receive_text()
        data = json.loads(message)
        
        if data.get('type') == 'authenticate' and data.get('token'):
            token = data['token']
            
            # Decode and validate token (you can use your existing authentication function)
            payload = decode_token(token)
            
        else:
            raise HTTPException(status_code=400, detail="Authentication required")
        
        # Handle communication (after successful authentication)
        while True:
            data = await ws.receive_text()
            
            await ws.send_text(f"Message received: {data}")

    except Exception as e:
        print(f"Error during WebSocket communication: {e}")
        await ws.close()

    finally:
        print("WebSocket connection closed")
        await ws.close()


# Static files for generated images
from fastapi.staticfiles import StaticFiles
from pathlib import Path

IMAGES_DIR = Path(__file__).parent / "generated_images"  # Ensure this path is correct

# Mount the static folder at the right path
app.mount("/api/demo_backend_v2/images", StaticFiles(directory=IMAGES_DIR), name="generated_images")

    
#Text to Image API endpoints
@app.post("/api/demo_backend_v2/generate", response_model=ImageGenerationResponse)
async def generate_image_endpoint(request: ImageGenerationRequest):
    """Endpoint for generating images from text"""
    return await generate_image(request)

@app.get("/api/demo_backend_v2/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}



# Delete face from collection and metadata table by photo
@app.post("/api/demo_backend_v2/delete_face_by_photo")
async def delete_face_by_photo_api(
    image: UploadFile = File(...),
    token: str = Depends(get_authorization_header),
    db: Session = Depends(get_db)
):
    if not image.filename:
        raise HTTPException(status_code=400, detail="Invalid file name.")

    # Save the uploaded file temporarily
    file_path = os.path.join(UPLOAD_FOLDER, image.filename)
    with open(file_path, "wb") as f:
        f.write(await image.read())

    try:
        result = delete_face_by_photo(image_path=file_path, db=db)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return result

    #api for search face in collection
@app.get("/api/demo_backend_v2/users")
def list_users(
    search: str = Query("", alias="search"), 
    db: Session = Depends(get_db)
):
    """
    List users, optionally filtered by name (case-insensitive search).
    """
    query = db.query(UserMetadata)
    if search:
        query = query.filter(UserMetadata.name.ilike(f"%{search}%"))
    users = query.all()
    return [
        {
            "name": user.name,
            "face_id": user.face_id,
            "image_url": user.image_url,  # Include image URL
        }
        for user in users
    ]
#api for delete user by face_id
@app.delete("/api/demo_backend_v2/users/{face_id}")
def delete_user(face_id: str, db: Session = Depends(get_db)):
    """
    Delete user from DB and Rekognition by face_id.
    """
    user = db.query(UserMetadata).filter(UserMetadata.face_id == face_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # Delete from Rekognition
    rekognition_client = get_rekognition_client_accountB()
    rekognition_client.delete_faces(
        CollectionId=FACE_COLLECTION_ID,
        FaceIds=[face_id]
    )

    # Delete from DB
    db.delete(user)
    db.commit()

    return {"message": f"User and face {face_id} deleted."}




# Handwritten Text converter

@app.post("/api/demo_backend_v2/extract-handwritten-text")
async def extract_handwritten_text_api(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted.")

    # Save file temporarily
    file_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Run extraction logic
    try:
        extracted_text = extract_handwritten_text(temp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temp file
        os.remove(temp_path)

    return {"extracted_text": extracted_text}



@app.post("/api/demo_backend_v2/estimate-calories")
async def estimate_calories(
    image: UploadFile = File(..., description="Food photo (jpg/png/webp)"),
    portion_hint: Optional[str] = Form(None, description="Optional text like '1 bowl', '2 slices', etc."),
):
    
    try:
        # Read the uploaded image
        img_bytes = await image.read()
        
        # Call the calorie estimation function
        result = analyze_calories_flow(img_bytes, portion_hint)

        # Ensure that fresh results are returned for each new request
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calorie estimation failed: {str(e)}")




@app.post("/api/demo_backend_v2/agentcore/invoke")
async def agentcore_invoke(payload: AgentCoreRequest):
    """
    Proxy to Bedrock AgentCore runtime with SigV4 signing.
    """
    try:
        from botocore.credentials import Credentials
        # Derive region from URL host; service is bedrock-agentcore
        parsed = urlparse(AGENTCORE_URL)
        host_parts = parsed.netloc.split(".")
        region = host_parts[1] if len(host_parts) > 1 else "us-east-1"
        service = "bedrock-agentcore"

        # Resolve AWS credentials (env vars, ~/.aws, or IAM role)
        session = boto3.Session()
        AWS_ACCESS_KEY_ID = "AKIA6ODUZ7QLQLM7ZNEN"
        AWS_SECRET_ACCESS_KEY = "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"

        credentials = Credentials(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        frozen = credentials.get_frozen_credentials()

        # Prepare and sign request
        body = json.dumps({"prompt": payload.prompt})
        headers = {"content-type": "application/json", "host": parsed.netloc}
        aws_request = AWSRequest(method="POST", url=AGENTCORE_URL, data=body, headers=headers)
        SigV4Auth(frozen, service, region).add_auth(aws_request)
        signed_headers = dict(aws_request.headers.items())

        # Send signed request
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(AGENTCORE_URL, headers=signed_headers, content=body)

        # Return JSON if possible; otherwise raw text
        try:
            content = resp.json()
        except Exception:
            content = {"raw": resp.text}
        return JSONResponse(status_code=resp.status_code, content=content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AgentCore proxy failed: {str(e)}")


@app.post("/api/demo_backend_v2/ai_concierge/ask")
def handle_ai_concierge(user_id: str = Body(...),question: str = Body(...)):
    try:
        answer = ai_concierge(user_id,question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


#eda setup

# Background task for cleaning up expired EDA sessions
async def cleanup_eda_sessions():
    """Background task to clean up expired EDA sessions every 5 minutes"""
    while True:
        try:
            file_manager.cleanup_expired_sessions()
            await asyncio.sleep(300)  # Run every 5 minutes
        except Exception as e:
            logger.error(f"Error cleaning up EDA sessions: {e}")
            await asyncio.sleep(300)

# All endpoints use Form parameters directly

# 0️⃣ Upload CSV and Create Session Route
@app.post("/api/demo_backend_v2/eda/upload")
async def upload_csv_session(file: UploadFile = File(...)):
    """Upload CSV file and create a session for EDA operations"""
    try:
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        # Read file content
        file_content = await file.read()
        
        # Create session
        session_id = file_manager.create_session(file_content, file.filename)
        
        # Get CSV info for the response
        csv_info = get_csv_info(session_id)
        
        return JSONResponse(content={
            "session_id": session_id,
            "message": "CSV uploaded successfully. Session created.",
            "csv_info": csv_info
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading CSV: {str(e)}")

# 1️⃣ Question Answering Route
@app.post("/api/demo_backend_v2/eda/qa")
async def question_answering(
    session_id: str = Form(...),
    question: str = Form(...)
):
    try:
        answer = ask_openai_from_session(session_id, question)
        return JSONResponse(content={"answer": answer})
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in QA: {str(e)}")

# 2️⃣ Distribution Graph Route
@app.post("/api/demo_backend_v2/eda/distribution")
async def distribution_graph(
    session_id: str = Form(...),
    column: str = Form(...)
):
    try:
        graphs = generate_graph_from_session(session_id, graph_type="dist", column=column)
        return JSONResponse(content={"graphs": graphs})
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in distribution graph: {str(e)}")

# 3️⃣ Time Series Graph Route
@app.post("/api/demo_backend_v2/eda/time")
async def time_series_graph(
    session_id: str = Form(...),
    column: str = Form(...)
):
    try:
        graphs = generate_graph_from_session(session_id, graph_type="time", column=column)
        return JSONResponse(content={"graphs": graphs})
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in time series graph: {str(e)}")

# 4️⃣ Correlation Heatmap Route
@app.post("/api/demo_backend_v2/eda/correlation")
async def correlation_graph(session_id: str = Form(...)):
    try:
        graphs = generate_graph_from_session(session_id, graph_type="corr")
        return JSONResponse(content={"graphs": graphs})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in correlation heatmap: {str(e)}")

# 5️⃣ Categorical Graph Route
@app.post("/api/demo_backend_v2/eda/categorical")
async def categorical_graph(
    session_id: str = Form(...),
    column: str = Form(...)
):
    try:
        graphs = generate_graph_from_session(session_id, graph_type="cat", column=column)
        return JSONResponse(content={"graphs": graphs})
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in categorical graph: {str(e)}")
