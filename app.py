from fastapi import FastAPI, UploadFile, File, HTTPException, Request, WebSocket, status, Query, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from pydantic import BaseModel
from typing import List
from werkzeug.utils import secure_filename
from datetime import datetime
import asyncio
from pathlib import Path
from fastapi.staticfiles import StaticFiles
import traceback
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

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
from face_recognigation.face_recognigation import add_face_to_collection, recognize_face
# In your app.py (or where you're using the models)
from face_recognigation._component.model import SessionLocal, UserMetadata
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


# # PDF data extraction imports
from pdf_data_extraction.app.config import TEMP_UPLOAD_DIR
from pdf_data_extraction.app.pdf_utils import extract_text_from_pdf, chunk_text
from pdf_data_extraction.app.embeddings import store_embeddings, query_embeddings, generate_answer
from pdf_data_extraction.app.models import UploadResponse, QuestionRequestPDF, AnswerResponse
from pdf_data_extraction.app.cleanup import cleanup_task

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

#Image search imports
import shutil
import tempfile
from img_search.embedder import Embedder
from img_search.s3_utils import S3Utils, ImageSearchRequest, ImageSearchResult
from img_search.vector_store import VectorStore

IMAGE_S3_BUCKET = "image2search"
FAISS_INDEX_FILE = "image_faiss_index.bin"
IMAGE_METADATA_FILE = "image_metadata.json"
s3_utils = S3Utils()
embedder = Embedder()
vector_store = VectorStore()
if not os.path.exists(FAISS_INDEX_FILE):
    s3_utils.download_file(IMAGE_S3_BUCKET, FAISS_INDEX_FILE, FAISS_INDEX_FILE)
if not os.path.exists(IMAGE_METADATA_FILE):
    s3_utils.download_file(IMAGE_S3_BUCKET, IMAGE_METADATA_FILE, IMAGE_METADATA_FILE)

vector_store.load_index_and_metadata(FAISS_INDEX_FILE, IMAGE_METADATA_FILE)
known_folder_names_lower = vector_store.get_unique_folder_names()

# Voice agent imports
from voice_agent.voice_agent import voice_websocket_endpoint

# # Initialize instances of your assistants
ddx_assistant = DDxAssistant()
pii_redactor = PiiRedactor()
pii_extractor = PiiExtractor()


class QuestionRequest(BaseModel):
    question: str

class PiiRequest(BaseModel):
    text: str


# Initialize FastAPI app
app = FastAPI()

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

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database setup
DATABASE_URL = "postgresql://postgres:demo.holbox.ai@database-1.carkqwcosit4.us-east-1.rds.amazonaws.com:5432/face_detection"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

#face_recognigation database setup
# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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

print(os.getenv("DATABASE_URL"))



@app.post("/api/demo_backend_v2/add_face")
async def add_user_face_api(image: UploadFile = File(...), name: str = Form(...), age: int = Form(None), gender: str = Form(None)):
    """
    API endpoint to add a face to the collection and store user data in RDS.
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="Invalid file name.")

    # Save uploaded image to a file
    file_path = os.path.join(UPLOAD_FOLDER, image.filename)
    with open(file_path, "wb") as f:
        f.write(await image.read())

    # Add face to collection and store metadata in RDS
    result = add_face_to_collection(file_path, name)  # This function adds the face to Rekognition

    if "face_id" not in result:
        raise HTTPException(status_code=500, detail="Face addition to Rekognition failed.")

    # Store user metadata in RDS
    db_session = SessionLocal()
    user_metadata = UserMetadata(
        face_id=result['face_id'],  # The face_id returned from Rekognition
        name=name,
        age=age,
        gender=gender,
        timestamp=datetime.now()
    )

    db_session.add(user_metadata)  # Add the user metadata to the session
    db_session.commit()  # Commit the transaction to save it to the database
    db_session.close()  # Close the session

    # Clean up the temporary file
    os.remove(file_path)

    return {"message": "Face added and metadata saved successfully", "face_id": result['face_id']}

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
    asyncio.create_task(cleanup_task())
    asyncio.create_task(cleanup_task_summ())


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

@app.post("/api/demo_backend_v2/pdf_data_extraction/upload_pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    pdf_id = f"{uuid.uuid4()}.pdf"
    pdf_path = os.path.join(TEMP_UPLOAD_DIR, pdf_id)

    with open(pdf_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Extract text & chunk
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(pages)

    # Store embeddings in vector DB
    store_embeddings(pdf_id, chunks)

    return UploadResponse(pdf_id=pdf_id, message="PDF uploaded and processed successfully")


@app.post("/api/demo_backend_v2/pdf_data_extraction/ask_question", response_model=AnswerResponse)
async def ask_question(req: QuestionRequestPDF):
    if not req.pdf_id:
        raise HTTPException(status_code=400, detail="PDF ID is required")
    
    retrieved_chunks = query_embeddings(req.pdf_id, req.question, top_k=3)
    if not retrieved_chunks:
        return AnswerResponse(answer="No relevant information found.", source_chunks=[])

    context_texts = [chunk["text"] for chunk in retrieved_chunks]

    answer = generate_answer(req.question, context_texts)

    return AnswerResponse(answer=answer, source_chunks=context_texts)

@app.get("/api/demo_backend_v2/")
async def root():
    return {"message": "Welcome to the FastAPI application!"}


@app.post("/api/demo_backend_v2/ddx")
async def ask_ddx(request: QuestionRequest):
    response = ddx_assistant.ask(request.question)
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
    print("Received Audio URL:", audio_url)

    if not audio_url:
        raise HTTPException(status_code=400, detail="Audio URL is required.")
    
    # Check if the audio is from the predefinedAudios folder
    if "predefinedAudios/" in audio_url:
        try:
            # Extract filename (e.g., predefinedAudios1.mp3 → predefinedAudios1)
            filename = os.path.basename(audio_url).rsplit('.', 1)[0]
            # Construct the expected S3 URI for summary.json
            summary_s3_uri = f"s3://{BUCKET_NAME}/health_scribe/predefinedAudios/{filename}_summary.json"
            print(f"Returning predefined summary URI: {summary_s3_uri}")
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
            print("Converted to S3 URL:", audio_url)

        PREDEFINED_PREFIX = f"s3://{BUCKET_NAME}/predefined/"
        if audio_url.startswith(PREDEFINED_PREFIX):
            print("Predefined audio detected. Fetching existing summary...")

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
        print("Starting transcription job:", job_name)
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
    response = ask_nl2sql(request.question)
    return {"answer": response}


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
@app.websocket("/api/demo_backend_v2/voice_agent/voice")
async def websocket_route(ws: WebSocket):
    await voice_websocket_endpoint(ws)


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


#Image search API endpoints
@app.post("/api/demo_backend_v2/image_search/search", response_model=List[ImageSearchResult])
def image_text_search(request: ImageSearchRequest):
    filter_by_folder = request.folder
    if not filter_by_folder:
        filter_by_folder = s3_utils.detect_folder_from_query(request.query, known_folder_names_lower)

    query_embedding = embedder.get_text_embedding(request.query)
    search_results = vector_store.search_images(query_embedding, k=request.k, filter_folder=filter_by_folder)

    results = []
    for distance, image_s3_path, folder_name in search_results:
        image_url = s3_utils.get_image_url_from_s3_path(image_s3_path)
        results.append(ImageSearchResult(
            distance=float(distance),
            folder=folder_name,
            s3_path=image_s3_path,
            image_url=image_url
        ))
    return results


@app.post("/api/demo_backend_v2/image_search/search_by_image", response_model=List[ImageSearchResult])
async def image_file_search(file: UploadFile = File(...), k: int = 5):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        query_embedding = embedder.get_image_embedding_from_file(tmp_path)
        search_results = vector_store.search_images(query_embedding, k=k)

        results = []
        for distance, image_s3_path, folder_name in search_results:
            image_url = s3_utils.get_image_url_from_s3_path(image_s3_path)
            results.append(ImageSearchResult(
                distance=float(distance),
                folder=folder_name,
                s3_path=image_s3_path,
                image_url=image_url
            ))
        return results
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.remove(tmp_path)
