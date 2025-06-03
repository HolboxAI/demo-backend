# Text-to-Video Generation API

This module provides an API for generating videos from text prompts using Amazon Bedrock's Nova model. It allows users to create short videos (6 seconds) based on text descriptions.

## Prerequisites

- Python 3.8+
- AWS Account with Bedrock access
- Required Python packages:
  ```bash
  pip install boto3 fastapi python-dotenv pydantic
  ```

## Environment Variables

Create a `.env` file in your project root with the following variables:

```env
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
S3_BUCKET_NAME=your_bucket_name
```

## API Endpoints

### 1. Generate Video

**Endpoint:** `POST /generate-video`

Generates a video from a text prompt.

**Request Body:**
```json
{
    "text": "A beautiful sunset over the ocean with waves crashing on the shore",
    "seed": 42  // Optional, defaults to 42
}
```

**Response:**
```json
{
    "job_id": "arn:aws:bedrock:region:account:async-invoke/...",
    "status": "processing",
    "message": "Video generation started successfully"
}
```

### 2. Check Video Status

**Endpoint:** `GET /video-status?job_id={invocation_arn}`

Checks the status of a video generation job.

**Query Parameters:**
- `job_id`: The full invocation ARN returned from the generate-video endpoint

**Response:**
```json
{
    "job_id": "arn:aws:bedrock:region:account:async-invoke/...",
    "status": "completed",
    "message": "Video generation completed successfully",
    "video_url": "https://s3.amazonaws.com/..."  // Only present when completed
}
```

## Usage Example

1. Start the video generation:
```bash
curl -X POST http://localhost:8000/generate-video \
  -H "Content-Type: application/json" \
  -d '{"text": "A beautiful sunset over the ocean"}'
```

2. Check the status:
```bash
curl -X GET "http://localhost:8000/video-status?job_id=arn:aws:bedrock:region:account:async-invoke/..."
```

3. Once completed, use the `video_url` to access the generated video.

## Video Generation Parameters

- **Duration:** 6 seconds
- **FPS:** 24
- **Resolution:** 1280x720
- **Seed:** Optional (0 to 2147483646)

## Error Handling

The API returns appropriate HTTP status codes and error messages:
- 400: Bad Request (invalid input)
- 500: Internal Server Error (AWS service errors)

## AWS Permissions Required

Ensure your AWS credentials have the following permissions:
- `bedrock:InvokeModel`
- `bedrock:StartAsyncInvoke`
- `bedrock:GetAsyncInvoke`
- `s3:PutObject`
- `s3:GetObject`

## Notes

- Videos are stored in the specified S3 bucket
- Generated video URLs expire after 1 hour
- The API uses asynchronous processing for video generation
- Status polling is recommended to check completion

## Limitations

- Maximum text prompt length: 512 characters
- Fixed video duration: 6 seconds
- Fixed resolution: 1280x720
- Fixed FPS: 24

## Troubleshooting

Common issues and solutions:
1. **ValidationException**: Check the text prompt length and format
2. **AccessDeniedException**: Verify AWS credentials and permissions
3. **ResourceNotFoundException**: Ensure the S3 bucket exists
4. **ServiceQuotaExceededException**: Check your AWS service quotas
