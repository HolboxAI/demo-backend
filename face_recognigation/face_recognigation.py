import boto3
import cv2
import os
from fastapi import HTTPException
import numpy as np

# AWS Rekognition Client
rekognition_client = boto3.client("rekognition", region_name="us-east-1")

# Configuration
FACE_COLLECTION_ID = "face-db-001"

def add_face_to_collection(image_path: str, external_image_id: str):
    """
    Add a face to the AWS Rekognition collection.
    
    Args:
        image_path: Path to the image file
        external_image_id: Name/ID to associate with the face
    
    Returns:
        dict: Response containing the face ID and other details
    """
    try:
        # Read and validate image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file or format")

        # Convert image to RGB (AWS Rekognition expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to JPEG format
        _, img_encoded = cv2.imencode('.jpg', image_rgb)
        image_bytes = img_encoded.tobytes()
        
        # Index the face in the collection
        response = rekognition_client.index_faces(
            CollectionId=FACE_COLLECTION_ID,
            Image={'Bytes': image_bytes},
            ExternalImageId=external_image_id,
            DetectionAttributes=['ALL']
        )
        
        if not response['FaceRecords']:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        return {
            "message": "Face added successfully",
            "face_id": response['FaceRecords'][0]['Face']['FaceId'],
            "external_image_id": external_image_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding face to collection: {str(e)}")

def recognize_face(image_path: str):
    """
    Recognize a face in the given image by comparing it with faces in the collection.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        dict: Response containing the recognized face details
    """
    try:
        # Read and validate image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file or format")

        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to JPEG format
        _, img_encoded = cv2.imencode('.jpg', image_rgb)
        image_bytes = img_encoded.tobytes()
        
        # Search for the face in the collection
        response = rekognition_client.search_faces_by_image(
            CollectionId=FACE_COLLECTION_ID,
            Image={'Bytes': image_bytes},
            MaxFaces=1,
            FaceMatchThreshold=70  # Minimum confidence threshold
        )
        
        if not response['FaceMatches']:
            return {
                "message": "No matching face found",
                "recognized": False
            }
        
        # Get the best match
        best_match = response['FaceMatches'][0]
        return {
            "message": "Face recognized",
            "recognized": True,
            "name": best_match['Face']['ExternalImageId'],
            "confidence": best_match['Similarity']
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recognizing face: {str(e)}")