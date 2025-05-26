
##  PII Redactor

```markdown
# PII Redactor Backend

This FastAPI backend provides an API for **redacting Personally Identifiable Information (PII)** from text using the `newberryai` package.

---

## Feature Description

The PII Redactor automatically detects and masks sensitive personal information such as names, emails, phone numbers, addresses, and other identifiers from input text to protect privacy.

---

## Setup

1. Clone the repository.
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

* **URL:** `/redact`

* **Method:** `POST`

* **Request Body:**

  ```json
  {
    "text": "Contact me at john.doe@example.com or call 123-456-7890."
  }
  ```

* **Response Body:**

  ```json
  {
    "redacted": "Contact me at [REDACTED] or call [REDACTED]."
  }
  ```

---

## Usage

Send a POST request to `/redact` with the text containing PII. The response returns the text with PII redacted.

---

## Notes

* Ensure the OpenAI API key is configured in `.env`.
* Test with Swagger UI at `http://localhost:8000/docs`.

---

## License

MIT License

````

---