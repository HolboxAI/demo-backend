from fastapi import FastAPI, UploadFile, File, HTTPException, Request, WebSocket, status, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from pydantic import BaseModel
from werkzeug.utils import secure_filename
from datetime import datetime
import asyncio
import traceback
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# #nl2sql imports
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

# # PDF data extraction imports
from pdf_data_extraction.app.config import TEMP_UPLOAD_DIR
from pdf_data_extraction.app.pdf_utils import extract_text_from_pdf, chunk_text
from pdf_data_extraction.app.embeddings import store_embeddings, query_embeddings, generate_answer
from pdf_data_extraction.app.models import UploadResponse, QuestionRequestPDF, AnswerResponse
from pdf_data_extraction.app.cleanup import cleanup_task

from ddx.ddx import DDxAssistant
from pii_redactor.redactor import PiiRedactor
from pii_extractor.extractor import PiiExtractor

#Text to video imports
from txt2vid.main import (
    VideoGenerationRequest,
    VideoGenerationResponse,
    generate_video,
    check_video_generation_status,
    get_video_url
)


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

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI application!"}

@app.post("/detect_faces")
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

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_task())


@app.post("/pdf_data_extraction/upload_pdf", response_model=UploadResponse)
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


@app.post("/pdf_data_extraction/ask_question", response_model=AnswerResponse)
async def ask_question(req: QuestionRequestPDF):
    if not req.pdf_id:
        raise HTTPException(status_code=400, detail="PDF ID is required")
    
    retrieved_chunks = query_embeddings(req.pdf_id, req.question, top_k=3)
    if not retrieved_chunks:
        return AnswerResponse(answer="No relevant information found.", source_chunks=[])

    context_texts = [chunk["text"] for chunk in retrieved_chunks]

    answer = generate_answer(req.question, context_texts)

    return AnswerResponse(answer=answer, source_chunks=context_texts)


@app.post("/ddx")
async def ask_ddx(request: QuestionRequest):
    response = ddx_assistant.ask(request.question)
    return {"answer": response}


@app.post("/redact")
async def redact_pii(request: PiiRequest):
    redacted_text = pii_redactor.redact(request.text)
    return {"redacted": redacted_text}


@app.post("/extract")
async def extract_pii(request: PiiRequest):
    extracted = pii_extractor.extract(request.text)
    return {"extracted": extracted}


# Healthscribe API endpoints
@app.post("/healthscribe/upload-audio")
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
    
@app.post("/healthscribe/question-ans")
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

@app.post("/healthscribe/start-transcription")
async def start_transcription_route(request: Request):
    global transcription_summary

    data = await request.json()
    audio_url = data.get('audioUrl')
    print("Received Audio URL:", audio_url)

    if not audio_url:
        raise HTTPException(status_code=400, detail="Audio URL is required.")

    try:
        BUCKET_NAME = "dax-healthscribe-v2"
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
            summary_s3_key = f"predefined/{summary_filename}"

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
    
@app.post("/nl2sql/ask")
async def ask_nl2sql_endpoint(request: QuestionRequest):
    response = ask_nl2sql(request.question)
    return {"answer": response}


#Virtual try on backend API endpoints
@app.post("/virtual-tryon/run", response_model=VirtualTryOnResponse)
async def virtual_tryon_run(request: VirtualTryOnRequest):
    """
    Process virtual try-on request (equivalent to handleProcess in React)
    """
    try:
        
        result = await handle_process(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Virtual try-on processing failed: {str(e)}")

@app.get("/virtual-tryon/status/{job_id}", response_model=StatusResponse)
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
    "/generate-video",
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
    "/video-status",
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
@app.websocket("/voice_agent/voice")
async def websocket_route(ws: WebSocket):
    await voice_websocket_endpoint(ws)
