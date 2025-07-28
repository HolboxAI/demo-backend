import boto3

import cv2
import os
from fastapi import HTTPException
import uuid
from sqlalchemy.orm import Session
from ._component.model import  UserMetadata
import numpy as np

FACE_COLLECTION_ID = "face-db-001"
S3_BUCKET_NAME=os.getenv("BUCKET_NAME") 

# AWS S3 client
s3_client = boto3.client('s3', region_name=os.getenv('AWS_DEFAULT_REGION'))
rekognition_client = boto3.client('rekognition', region_name=os.getenv('AWS_DEFAULT_REGION'))


# Utility: Create AccountB Rekognition client
def get_rekognition_client_accountB():
    return boto3.client(
        "rekognition",
        region_name="us-east-1",
        aws_access_key_id=os.environ["AWS_ACCOUNTB_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_ACCOUNTB_SECRET_ACCESS_KEY"],
        # aws_session_token=os.environ.get("AWS_ACCOUNTB_SESSION_TOKEN"), # if you use temporary creds
    )




def add_face_to_collection(image_bytes, external_image_id):
    try:
        # Proceed with face detection as normal
        rekognition_client = get_rekognition_client_accountB()
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

    
# def add_face_to_collection(image_file, external_image_id):
#     try:
#         # Read image from the file object into memory as bytes
#         image_bytes = image_file.read()

#         # Convert the image bytes to a numpy array
#         nparr = np.frombuffer(image_bytes, np.uint8)

#         # Decode the numpy array to an image (OpenCV format)
#         image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         if image is None:
#             raise HTTPException(status_code=400, detail="Invalid image file or format")

#         # Proceed with face detection as normal
#         rekognition_client = get_rekognition_client_accountB()
#         response = rekognition_client.index_faces(
#             CollectionId=FACE_COLLECTION_ID,
#             Image={'Bytes': image_bytes},
#             ExternalImageId=external_image_id,
#             DetectionAttributes=['ALL']
#         )

#         if not response['FaceRecords']:
#             raise HTTPException(status_code=400, detail="No face detected in the image")

#         return {
#             "message": "Face added successfully",
#             "face_id": response['FaceRecords'][0]['Face']['FaceId'],
#             "external_image_id": external_image_id
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error adding face to collection: {str(e)}")
# Function to add face to Rekognition collection (corrected to handle bytes)
# def add_face_to_collection(image_bytes, external_image_id):
#     try:
#         # Add face to Rekognition collection (passing the raw bytes directly)
#         response = rekognition_client.index_faces(
#             CollectionId="face-db-001",  # Replace with your collection ID
#             Image={'Bytes': image_bytes},  # Pass the image bytes directly here
#             ExternalImageId=external_image_id,  # Use external image ID as metadata
#             DetectionAttributes=['ALL']
#         )

#         if not response['FaceRecords']:
#             raise HTTPException(status_code=400, detail="No face detected in the image")

#         face_id = response['FaceRecords'][0]['Face']['FaceId']  # Extract the face ID from the response
#         return {"face_id": face_id}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error adding face to collection: {str(e)}")


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
    


def delete_face_by_photo(image_path: str, db: Session):
    """
    Recognize face from image, delete from Rekognition collection, and delete user metadata from DB.
    """
    # Step 1: Recognize face
    result = recognize_face(image_path)
    if not result.get("recognized") or "face_id" not in result:
        raise HTTPException(status_code=404, detail="No matching face found.")

    face_id = result["face_id"]

    # Step 2: Delete from Rekognition collection
    rekognition_client = get_rekognition_client_accountB()
    rekognition_client.delete_faces(
        CollectionId=FACE_COLLECTION_ID,
        FaceIds=[face_id]
    )

    # Step 3: Delete user metadata from DB
    user_metadata = db.query(UserMetadata).filter(UserMetadata.face_id == face_id).first()
    if user_metadata:
        db.delete(user_metadata)
        db.commit()
        message = f"Face and user data deleted successfully (face_id={face_id})"
    else:
        message = f"Face deleted from collection, but no user metadata found for face_id={face_id}"

    return {"message": message}

# Function to upload the image to S3 using boto3 (synchronous)
# def upload_to_s3(image_bytes, s3_key):
#     try:
#         # Upload the image to S3 bucket
#         response = s3_client.put_object(
#             Bucket=S3_BUCKET_NAME,
#             Key=s3_key,
#             Body=image_bytes,
#             ContentType='image/jpeg',  # Adjust the content type if necessary
#         )

#         # Check if the upload was successful by checking the HTTPStatusCode
#         if response['ResponseMetadata']['HTTPStatusCode'] == 200:
#             print(f"Successfully uploaded {s3_key} to S3.")
#         else:
#             raise HTTPException(status_code=500, detail=f"Failed to upload image to S3. Response: {response}")

#         # Generate the URL to access the uploaded image
#         image_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
#         return image_url
#     except Exception as e:
        
#         raise HTTPException(status_code=500, detail=f"Error uploading to S3: {str(e)}")

def upload_to_s3(image_bytes, s3_key):
    try:
        response = s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=image_bytes,
            ContentType='image/jpeg',
            
        )

        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            # US East 1 format:
            image_url = f"https://s3.us-east-1.amazonaws.com/{S3_BUCKET_NAME}/{s3_key}"
            return image_url
        else:
            raise HTTPException(status_code=500, detail=f"Failed to upload image to S3. Response: {response}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading to S3: {str(e)}")


# Function to add face to Rekognition and upload image to S3
async def add_face_and_upload(image, name: str, background_tasks):
    # Read the image content into memory
    image_bytes = await image.read()

    # Generate a unique key for the image in the S3 bucket
    s3_key = f"face_rekognition/face/{uuid.uuid4()}.jpg"  # Save image under the 'face_rekognition/face/' folder

    # Upload the image to S3 and get the URL
    image_url = upload_to_s3(image_bytes, s3_key)  # Use boto3 to upload the image and get the URL

    # Add the face to Rekognition collection
    result = add_face_to_collection(image_bytes, name)  # Call Rekognition to add face

    # Check if face_id is returned from Rekognition
    if "face_id" not in result:
        raise HTTPException(status_code=500, detail="Face addition to Rekognition failed.")

    # Return the face_id from Rekognition and the image URL from S3
    return result['face_id'], image_url