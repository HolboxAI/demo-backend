# PDF-QA-Backend

This is the backend service for the PDF Q&A system, responsible for processing uploaded PDF documents, extracting their content, and answering user questions using Retrieval-Augmented Generation (RAG).

## Overview

The backend provides API endpoints that allow users to:

- Upload PDF documents.
- Extract and embed their content for semantic search.
- Ask natural language questions about the document and receive contextual answers.

## Features

- **Upload PDF files** and automatically extract and chunk text content.
- **Embed and store chunks** for semantic retrieval using a vector database.
- **Answer user questions** using a language model based on relevant chunks.

## Response Format

When a PDF is uploaded and processed successfully:

```json
{
  "pdf_id": "123e4567-e89b-12d3-a456-426614174000.pdf",
  "message": "PDF uploaded and processed successfully"
}
```

When a question is asked:

```json
{
  "answer": "The document discusses the impact of AI in healthcare.",
  "source_chunks": [
    "AI has significantly improved diagnostic accuracy in recent years...",
    "Machine learning models help analyze medical imaging..."
  ]
}
```

- `pdf_id`: A unique identifier for the uploaded PDF.

- `answer`: Natural-language answer to the user’s question.

- `source_chunks`: Text snippets from the PDF used to generate the answer.


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

### `/pdf_data_extraction/upload_pdf` (POST)

#### Description

Uploads a PDF file, extracts and embeds the content.

#### Request Body

- **file (.pdf)** : The PDF file (multipart/form-data)

#### Example Request

```bash
curl -X 'POST' \
  'http://localhost:8000/pdf_data_extraction/upload_pdf' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@document.pdf'

```

#### Example Response

```json
{
  "pdf_id": "f8263c14-3d63-4cf6-8c0e-f6ad6f6b4f8b.pdf",
  "message": "PDF uploaded and processed successfully"
}

```

### `/pdf_data_extraction/ask_question` (POST)

#### Description

Asks a natural language question about the uploaded PDF document.

#### Request Body
```json
{
  "pdf_id": "f8263c14-3d63-4cf6-8c0e-f6ad6f6b4f8b.pdf",
  "question": "What is the main topic of the document?"
}
```


#### Example Request

```bash
curl -X POST http://localhost:8000/pdf_data_extraction/ask_question \
  -H "Content-Type: application/json" \
  -d '{
        "pdf_id": "your_uploaded_pdf_id.pdf",
        "question": "What is the main topic of the document?"
      }'

```

#### Example Response

```json
{
  "answer": "The document discusses climate change and global policy impact.",
  "source_chunks": [
    "Climate change has become a global concern...",
    "Governments are implementing new policies..."
  ]
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
