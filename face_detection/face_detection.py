import os
import cv2
import boto3
import tempfile
import shutil

# AWS Rekognition Client
rekognition_client = boto3.client("rekognition", region_name="us-east-1")

# Configuration
FACE_COLLECTION_ID = "face-db-001"

def process_video_frames(video_path: str, max_frames: int = 20):
    """
    Extracts up to `max_frames` evenly spaced frames from a video and checks each with AWS Rekognition.
    """
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return {"error": "Could not open video file."}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total Frames: {total_frames}, FPS: {fps}")

    frame_interval = max(total_frames // max_frames, 1)
    frame_positions = range(0, total_frames, frame_interval)[:max_frames]

    frames_dir = tempfile.mkdtemp()
    detected_faces = []

    try:
        for frame_pos in frame_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            success, frame = cap.read()
            if not success:
                continue

            timestamp = round(frame_pos / fps, 2)
            print(f"Processing frame at position {frame_pos} ({timestamp}s)")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, img_encoded = cv2.imencode(".jpg", frame_rgb)
            img_bytes = img_encoded.tobytes()

            try:
                detect_response = rekognition_client.detect_faces(
                    Image={"Bytes": img_bytes},
                    Attributes=["DEFAULT"]
                )

                if detect_response.get("FaceDetails"):
                    print("Face detected")

                    search_response = rekognition_client.search_faces_by_image(
                        CollectionId=FACE_COLLECTION_ID,
                        Image={"Bytes": img_bytes},
                        MaxFaces=1
                    )

                    if search_response.get("FaceMatches"):
                        matched_face = search_response["FaceMatches"][0]["Face"]
                        external_image_id = matched_face.get("ExternalImageId", "Unknown")
                        print(f"Match found: {external_image_id}")

                        detected_faces.append({
                            "timestamp": timestamp,
                            "external_image_id": external_image_id
                        })
                    else:
                        print("No match found.")
                else:
                    print("No face detected.")
            except Exception as e:
                print(f"Error with frame {frame_pos}: {str(e)}")
    finally:
        cap.release()
        shutil.rmtree(frames_dir)  # Clean up temp directory

    print("Processing complete.")
    return detected_faces
