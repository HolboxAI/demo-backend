
---

##  DDx Assistant

```markdown
# DDx Assistant Backend

This project provides a FastAPI backend endpoint for the **Differential Diagnosis Assistant (DDx Assistant)** using the `newberryai` package.

---

## Feature Description

The DDx Assistant is an AI-powered tool that helps users by answering medical diagnostic questions based on symptoms and medical data. It supports interactive querying to provide differential diagnosis suggestions.

---

## Setup

1. Clone the repository.
2. Create a `.env` file with your OpenAI API key:

```

OPENAI\_API\_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

````

3. Install dependencies:

```bash
pip install -r requirements.txt
````

4. Run the FastAPI server:

   ```bash
   uvicorn app:app --reload
   ```

---

## API Endpoint

* **URL:** `/ddx`

* **Method:** `POST`

* **Request Body:**

  ```json
  {
    "question": "What could cause persistent cough and fever?"
  }
  ```

* **Response Body:**

  ```json
  {
    "answer": "Possible causes include pneumonia, tuberculosis, bronchitis, etc."
  }
  ```

---

## Usage

Send a POST request to `/ddx` with a JSON payload containing the medical question. The response contains the AI-generated differential diagnosis answer.

---

## Notes

* Make sure the OpenAI API key is set correctly in `.env`.
* You can test the API using Swagger UI at `http://localhost:8000/docs`.

---

## License

MIT License

````

