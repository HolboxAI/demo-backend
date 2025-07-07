# Image Search API

This module provides an API for searching images using both text queries and image similarity. It uses OpenAI's CLIP model for generating embeddings and FAISS for efficient vector similarity search. Images are stored in AWS S3 and organized by folders.

## Prerequisites

- Python 3.8+
- AWS Account with S3 access
- Required Python packages:
  ```bash
  pip install torch transformers pillow faiss-cpu boto3 numpy pydantic
  ```

## Environment Variables

Create a `.env` file in your project root with the following variables:

```env
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
```

## Components

### 1. Embedder
- Uses OpenAI's CLIP model (`openai/clip-vit-base-patch32`)
- Generates embeddings for both text and images
- Supports GPU acceleration when available

### 2. S3Utils
- Manages S3 operations (upload, download, list objects)
- Handles image loading from S3 paths
- Generates public and presigned URLs for images

### 3. VectorStore
- Uses FAISS for efficient similarity search
- Maintains metadata for images (S3 paths, folder names)
- Supports filtered search by folder

## API Endpoints

### 1. Text-Based Image Search

**Endpoint:** `POST /api/demo_backend_v2/image_search/search`

Search for images using text queries. The system automatically detects folder names from the query or you can specify a folder explicitly.

**Request Body:**
```json
{
    "query": "beautiful sunset over mountains",
    "k": 5,  // Optional, number of results (default: 5)
    "folder": ""  // Optional, filter by specific folder
}
```

**Response:**
```json
[
    {
        "distance": 0.234,
        "folder": "tshirt",
        "s3_path": "s3://image2search/nature/sunset1.jpg",
        "image_url": "https://image2search.s3.amazonaws.com/nature/sunset1.jpg"
    },
    {
        "distance": 0.456,
        "folder": "tshirt",
        "s3_path": "s3://image2search/nature/mountain_view.jpg",
        "image_url": "https://image2search.s3.amazonaws.com/nature/mountain_view.jpg"
    }
]
```

### 2. Image-Based Similarity Search

**Endpoint:** `POST /api/demo_backend_v2/image_search/search_by_image`

Search for similar images by uploading an image file.

**Request Parameters:**
- `file`: Image file (multipart/form-data)
- `k`: Number of results (query parameter, default: 5)

**Response:**
```json
[
    {
        "distance": 0.123,
        "folder": "animals",
        "s3_path": "s3://image2search/animals/cat1.jpg",
        "image_url": "https://image2search.s3.amazonaws.com/animals/cat1.jpg"
    },
    {
        "distance": 0.234,
        "folder": "animals",
        "s3_path": "s3://image2search/animals/dog2.jpg",
        "image_url": "https://image2search.s3.amazonaws.com/animals/dog2.jpg"
    }
]
```

## Usage Examples

### 1. Text-based search:
```bash
curl -X POST http://localhost:8000/api/demo_backend_v2/image_search/search \
  -H "Content-Type: application/json" \
  -d '{"query": "red car in the city", "k": 3}'
```

### 2. Image-based search:
```bash
curl -X POST http://localhost:8000/api/demo_backend_v2/image_search/search_by_image \
  -F "file=@/path/to/your/image.jpg" \
  -F "k=5"
```

## Image Organization

Example:
Images are stored in S3 with the following structure:
```
s3://image2search/
├── nature/
│   ├── sunset1.jpg
│   ├── mountain_view.jpg
│   └── forest.jpg
├── animals/
│   ├── cat1.jpg
│   ├── dog2.jpg
│   └── bird.jpg
└── vehicles/
    ├── car1.jpg
    └── truck.jpg
```

## Search Features

### Automatic Folder Detection
The system can automatically detect folder names from text queries:
- Query: "show me nature photos" → Automatically searches in "nature" folder
- Query: "find animal pictures" → Automatically searches in "animals" folder

### Distance Scoring
- Lower distance values indicate higher similarity
- Distance is calculated using L2 (Euclidean) distance in the embedding space
- Typical range: 0.0 (identical) to 2.0+ (very different)

### Supported Image Formats
- PNG
- JPG/JPEG
- GIF
- BMP
- TIFF

## Configuration

The module uses the following configuration:
- **S3 Bucket:** `image2search`
- **FAISS Index File:** `image_faiss_index.bin`
- **Metadata File:** `image_metadata.json`
- **CLIP Model:** `openai/clip-vit-base-patch32`

## Error Handling

The API returns appropriate HTTP status codes and error messages:
- 400: Bad Request (invalid file format, missing parameters)
- 404: Not Found (image not found)
- 500: Internal Server Error (S3 errors, model errors)

## AWS Permissions Required

Ensure your AWS credentials have the following permissions:
- `s3:GetObject`
- `s3:PutObject`
- `s3:ListBucket`
- `s3:DeleteObject`

## Performance Considerations

- FAISS index and metadata are loaded at startup
- Index is stored locally for faster access
- GPU acceleration is used when available
- S3 operations are optimized with proper pagination

## Limitations

- Maximum file size for image uploads: Depends on FastAPI configuration
- CLIP model supports images up to 224x224 (automatically resized)
- Search results are limited by the `k` parameter (maximum recommended: 100)

## Troubleshooting

Common issues and solutions:

1. **Model Loading Error**: Ensure sufficient memory and proper PyTorch installation
2. **S3 Access Denied**: Verify AWS credentials and bucket permissions
3. **FAISS Index Not Found**: Check if index files exist in S3 and can be downloaded
4. **Image Processing Error**: Verify image format and file integrity
5. **Out of Memory**: Reduce batch size or use CPU instead of GPU

## Maintenance

### Updating the Index
To add new images to the search index:
1. Upload images to S3 in organized folders
2. Run the index building process to update FAISS index
3. Upload updated index and metadata files to S3

### Monitoring
Monitor the following metrics:
- Search response times
- S3 operation success rates
- Model inference latency
- Memory usage
