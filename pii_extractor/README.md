
---

##  PII Extractor

```markdown
# PII Extractor Backend

This FastAPI backend exposes an API to **extract Personally Identifiable Information (PII)** from text using the `newberryai` package.

---

## Feature Description

The PII Extractor detects and extracts sensitive information such as phone numbers, emails, names, addresses, etc., from input text, facilitating data processing and compliance.

---

## Setup

1. Clone the repo.
2. Create a `.env` file with your OpenAI API key:

````

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

* **URL:** `/extract`

* **Method:** `POST`

* **Request Body:**

  ```json
  {
    "text": "My phone number is (123) 456-7890 and my email is jane@example.com."
  }
  ```

* **Response Body:**

  ```json
  {
    "extracted": {
      "phone": "(123) 456-7890",
      "email": "jane@example.com"
    }
  }
  ```

---

## Usage

Send a POST request to `/extract` with text input. The response contains the extracted PII entities in JSON format.

---

## Notes

* The extraction keys depend on the underlying AI model.
* Configure your OpenAI API key in `.env`.
* Test easily with Swagger UI at `http://localhost:8000/docs`.

---

## License

MIT License

```

---

