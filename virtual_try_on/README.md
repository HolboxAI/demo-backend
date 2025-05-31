# Virtual Try-On Backend

This FastAPI backend exposes an API to **try garments on models virtually** using the Fashn API. It allows you to upload a model photo and a garment image, and returns a composite try-on output.

---

## Feature Description

This service enables **AI-powered fashion try-on**. Users can upload base64-encoded images of a person (model) and a clothing item (garment), and the system returns a photorealistic output showing how the garment would look on the model.

Ideal for fashion e-commerce, virtual shopping, and styling apps.

---

## Setup

1. Clone the repo.
2. Create a `.env` file in the root directory with the following:

    ````env
    FASHN_API_URL=https://your-fashn-api-endpoint.com
    FASHN_AUTH_KEY=your_auth_key_here
    ````

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the FastAPI server:

    ```bash
    uvicorn app:app --reload
    ```

---

## API Endpoints

### 1. `/virtual-tryon/run`

- **Method:** `POST`
- **Description:** Starts a virtual try-on process.

- **Request Body:**

    ```json
    {
      "model_image": "base64_string_of_model_image",
      "garment_image": "base64_string_of_garment_image",
      "category": "tops"
    }
    ```

- **Response Body:**

    ```json
    {
      "id": "job_id_123"
    }
    ```

---

### 2. `/virtual-tryon/status/{job_id}`

- **Method:** `GET`
- **Description:** Fetch the status of the virtual try-on job.

- **Response Body (on success):**

    ```json
    {
      "status": "completed",
      "output": [
        "https://cdn.fashn.ai/generated_output_1.jpg"
      ]
    }
    ```

- **Possible statuses:**
  - `processing`
  - `completed`
  - `failed`

---

## Usage

1. Send a POST request to `/virtual-tryon/run` with base64 images.
2. Receive a job ID in response.
3. Poll `/virtual-tryon/status/{job_id}` every few seconds to check processing status.
4. Once completed, you’ll get URLs to the generated try-on images.

---

## Notes

- Uses Fashn's third-party AI API under the hood.
- All image inputs must be base64-encoded strings.
- Configure the `.env` with valid Fashn API credentials.
- Easy testing available via Swagger UI: `http://localhost:8000/docs`.

---

## License

MIT License
