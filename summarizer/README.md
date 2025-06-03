# PDF Summarization Backend

This is the backend service for the PDF Summarization system, responsible for processing uploaded PDF documents, extracting their content, and generating concise summaries using OpenAI's language models.

## Overview

The backend provides API endpoints that allow users to:

- Upload PDF documents.
- Extract their full text content.
- Generate AI-based summaries of the document content.

## Features

- **Upload PDF files** and temporarily store them on the server.
- **Extract full text** content from PDFs using PyMuPDF.
- **Generate document summaries** using OpenAI's GPT models.
- Simple and lightweight with no embedding or vector database.

## Response Format

When a PDF is uploaded successfully:

```json
{
  "pdf_id": "123e4567-e89b-12d3-a456-426614174000.pdf",
  "message": "PDF uploaded successfully"
}


When a summary is requested

```json
{
  "summary": "This document discusses the recent advances in AI technology, focusing on natural language processing..."
}

```

- `pdf_id`: A unique identifier for the uploaded PDF.

- `summary`: The AI-generated summary text of the PDF content.



## Installation

### Prerequisites

1. Python 3.7 or above

### Dependencies

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains the following packages:

- `openai`: For Embeddings models and LLM.
- `PyMuPDF`: For Processing the data of PDF files.
- `fastapi`: Web framework for building APIs.
- `uvicorn`: ASGI server for running the FastAPI app.
- `python-multipart`: To handle file uploads in FastAPI.

### Requirements.txt

```txt
uvicorn
python-multipart
uvicorn[standard]
PyMuPDF
openai
faiss-cpu
numpy
python-dotenv
fastapi
```


## Running the App

1. To start the backend server, run the following command:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

```

This will start the FastAPI server on port `8000` and enable auto-reloading during development.

2. The API will be accessible at `http://localhost:8000`.

## API Endpoint

### `/summarizer/upload_pdf` (POST)

#### Description

Upload a PDF file, which will be temporarily stored for processing.

#### Request Body

- **file (.pdf)** : The PDF file (multipart/form-data)

#### Example Request

```bash
curl -X POST "http://localhost:8000/upload_pdf" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf"

```

#### Example Response

```json
{
  "pdf_id": "f8263c14-3d63-4cf6-8c0e-f6ad6f6b4f8b.pdf",
  "message": "PDF uploaded successfully"
}

```

### `/summarizer/get_summary` (POST)

#### Description

Generate and retrieve an AI summary for the uploaded PDF.

#### Request Body
```json
{
  "pdf_id": "f8263c14-3d63-4cf6-8c0e-f6ad6f6b4f8b.pdf"
}

```


#### Example Request

```bash
curl -X POST "http://localhost:8000/get_summary" \
  -H "Content-Type: application/json" \
  -d '{"pdf_id": "your_uploaded_pdf_id"}'

```

#### Example Response

```json
{
  "summary": "This document explores the fundamentals of quantum computing, including key algorithms and potential applications."
}

```

## Development

### Running the App Locally

- Install dependencies using `pip install -r requirements.txt`.
- Start the FastAPI server using `uvicorn`:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

This will make the backend available at `http://localhost:8000`.



## License

This project is licensed under the MIT License.

---
