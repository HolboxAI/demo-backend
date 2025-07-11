import boto3
import cv2
import os
from fastapi import HTTPException

FACE_COLLECTION_ID = "face-db-001"

# Utility: Create AccountB Rekognition client
def get_rekognition_client_accountB():
    return boto3.client(
        "rekognition",
        region_name="us-east-1",
        aws_access_key_id=os.environ["AWS_ACCOUNTB_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_ACCOUNTB_SECRET_ACCESS_KEY"],
        # aws_session_token=os.environ.get("AWS_ACCOUNTB_SESSION_TOKEN"), # if you use temporary creds
    )

def add_face_to_collection(image_path: str, external_image_id: str):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file or format")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, img_encoded = cv2.imencode('.jpg', image_rgb)
        image_bytes = img_encoded.tobytes()

        rekognition_client = get_rekognition_client_accountB()  # Use AccountB creds
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
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file or format")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, img_encoded = cv2.imencode('.jpg', image_rgb)
        image_bytes = img_encoded.tobytes()

        rekognition_client = get_rekognition_client_accountB()  # Use AccountB creds
        response = rekognition_client.search_faces_by_image(
            CollectionId=FACE_COLLECTION_ID,
            Image={'Bytes': image_bytes},
            MaxFaces=1,
            FaceMatchThreshold=70
        )

        if not response['FaceMatches']:
            return {
                "message": "No matching face found",
                "recognized": False
            }

        best_match = response['FaceMatches'][0]
        return {
            "message": "Face recognized",
            "recognized": True,
            "face_id": best_match['Face']['FaceId'],
            "name": best_match['Face']['ExternalImageId'],
            "confidence": best_match['Similarity']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recognizing face: {str(e)}")
