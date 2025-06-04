# Face Recognition Module

This module provides face recognition capabilities using AWS Rekognition, allowing users to add faces to a collection and recognize faces from uploaded images.

## Overview

The face recognition module offers two main functionalities:
1. Adding faces to an AWS Rekognition collection with associated names/IDs
2. Recognizing faces by comparing them against the existing collection

## Features

* **Add faces to collection**: Upload an image with a name to store in the face database
* **Recognize faces**: Upload an image to identify if it matches any face in the collection
* **Confidence scoring**: Get similarity scores for face matches
* **Error handling**: Robust error handling for invalid images and failed operations

## Response Formats

### Add Face Response
```json
{
    "message": "Face added successfully",
    "face_id": "1234567890",
    "external_image_id": "person_name"
}
```

### Recognize Face Response
```json
{
    "message": "Face recognized",
    "recognized": true,
    "name": "person_name",
    "confidence": 95.5
}
```

## Installation

### Prerequisites

1. Python 3.7 or above
2. AWS Rekognition setup with a face collection created
3. AWS credentials configured

### Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

Required packages:
* `opencv-python`: For image processing
* `boto3`: AWS SDK for Python
* `fastapi`: Web framework
* `python-multipart`: For handling file uploads
* `numpy`: For image array operations

### Requirements.txt
```txt
opencv-python
boto3
fastapi
python-multipart
numpy
```

### AWS Configuration

1. Configure AWS credentials:
```bash
aws configure
```

2. Ensure you have:
   - Valid AWS credentials
   - Appropriate IAM permissions for Rekognition
   - A face collection created in Rekognition

## API Endpoints

### `/add_face` (POST)

#### Description
Adds a face to the AWS Rekognition collection.

#### Request Body
* **image**: The image file to be uploaded (JPG, PNG)
* **name**: The name/ID to associate with the face

#### Example Request
```bash
curl -X 'POST' \
  'http://localhost:8000/add_face' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@path_to_image/image.jpg' \
  -F 'name=person_name'
```

### `/recognize_face` (POST)

#### Description
Recognizes a face by comparing it with faces in the collection.

#### Request Body
* **image**: The image file to be recognized (JPG, PNG)

#### Example Request
```bash
curl -X 'POST' \
  'http://localhost:8000/recognize_face' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@path_to_image/image.jpg'
```

## Usage Guidelines

1. **Adding Faces**:
   - Use clear, front-facing images
   - Ensure good lighting conditions
   - Use consistent image formats (JPG recommended)
   - Provide unique names for each person

2. **Recognizing Faces**:
   - Use images with clear, visible faces
   - Similar lighting conditions to training images
   - Minimum confidence threshold is set to 70%

## Error Handling

The module handles various error cases:
* Invalid image formats
* No face detected in image
* AWS Rekognition service errors
* Invalid file uploads

## Development

### Running Tests
```bash
# Add test commands here when available
```

### Best Practices
1. Always validate images before processing
2. Handle AWS credentials securely
3. Implement proper error handling
4. Use appropriate image formats and sizes

## License

This project is licensed under the MIT License.

---