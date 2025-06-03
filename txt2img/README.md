# Text-to-Image Generation API

This module provides an API for generating images from text prompts using Amazon Bedrock's Titan Image Generator v2 model. It allows users to create high-quality images based on text descriptions.

## Prerequisites

- Python 3.8+
- AWS Account with Bedrock access
- Required Python packages:
  ```bash
  pip install boto3 fastapi python-dotenv pydantic
  ```

## Environment Variables

Create a `.env` file in your project root with the following variables:

```
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
```

## API Endpoints

### 1. Generate Image
**Endpoint:** POST /generate

Generates an image from a text prompt.

**Request Body:**
```json
{
    "text": "A beautiful sunset over the ocean with waves crashing on the shore",
    "width": 1024,          // Optional, defaults to 1024
    "height": 1024,         // Optional, defaults to 1024
    "number_of_images": 1,  // Optional, defaults to 1
    "cfg_scale": 8,         // Optional, defaults to 8
    "seed": 42,             // Optional, defaults to 42
    "quality": "standard"   // Optional, defaults to "standard"
}
```

**Response:**
```json
{
    "message": "Images generated successfully",
    "images": ["http://localhost:8001/images/<filename>.png"],
    "local_path": "generated_images"
}
```

## Usage Example

Generate an image:
```bash
curl -X POST "http://localhost:8000/api/demo_backend_v2/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "A beautiful sunset over the ocean",
    "width": 1024,
    "height": 1024,
    "number_of_images": 1,
    "cfg_scale": 8,
    "seed": 42,
    "quality": "standard"
}'
```

## Image Generation Parameters
- Resolution: 512x512 to 1024x1024
- Quality: standard or premium
- CFG Scale: 1 to 20 (default: 8)
- Seed: Optional (0 to 2147483646)
- Number of Images: 1 to 4

## Error Handling
The API returns appropriate HTTP status codes and error messages:

- 400: Bad Request (invalid input)
- 500: Internal Server Error (AWS service errors)

## AWS Permissions Required
Ensure your AWS credentials have the following permissions:

- bedrock:InvokeModel
- bedrock:GetModelInvocationLoggingConfiguration

## Notes
- Images are stored in the local `generated_images` directory
- Each image is saved with a unique UUID filename
- Images can be accessed via the returned URLs


## Image Storage

- Generated images are saved in the `generated_images` directory
- Each image is saved with a unique UUID filename
- Images can be accessed via the returned URLs or directly from the `generated_images` directory

## Limitations

- Maximum text prompt length: 512 characters
- Maximum image dimensions: 1024x1024
- Maximum number of images per request: 4
- Supported image format: PNG

## Troubleshooting

Common issues and solutions:

1. **ValidationException:**
   - Check the text prompt length and format
   - Verify image dimensions are within limits

2. **AccessDeniedException:**
   - Verify AWS credentials and permissions
   - Check if Bedrock service is enabled in your region

3. **ServiceQuotaExceededException:**
   - Check your AWS service quotas
   - Reduce the number of concurrent requests

4. **Image Generation Failures:**
   - Verify the prompt is clear and well-formed
   - Try different seed values
   - Adjust CFG scale if needed

