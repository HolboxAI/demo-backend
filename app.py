from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from face_detection.face_detection import process_video_frames  # Importing face detection logic
import os


import uuid
from pdf_data_extraction.app.config import TEMP_UPLOAD_DIR
from pdf_data_extraction.app.pdf_utils import extract_text_from_pdf, chunk_text
from pdf_data_extraction.app.embeddings import store_embeddings, query_embeddings, generate_answer
from pdf_data_extraction.app.models import UploadResponse, QuestionRequest, AnswerResponse
from pdf_data_extraction.app.cleanup import cleanup_task
import asyncio

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

@app.post("pdf_data_extraction/upload_pdf", response_model=UploadResponse)
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


@app.post("pdf_data_extraction/ask_question", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    if not req.pdf_id:
        raise HTTPException(status_code=400, detail="PDF ID is required")
    
    retrieved_chunks = query_embeddings(req.pdf_id, req.question, top_k=3)
    if not retrieved_chunks:
        return AnswerResponse(answer="No relevant information found.", source_chunks=[])

    context_texts = [chunk["text"] for chunk in retrieved_chunks]

    answer = generate_answer(req.question, context_texts)

    return AnswerResponse(answer=answer, source_chunks=context_texts)
