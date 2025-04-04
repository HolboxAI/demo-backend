from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from face_detection import process_video_frames  # Importing face detection logic
import os

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
