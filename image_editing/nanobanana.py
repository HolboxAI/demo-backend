import os
import base64
from pydantic import BaseModel
from google import genai
from google.genai.types import HttpOptions, Part, Content, GenerateContentConfig, SafetySetting

class EditImageResponse(BaseModel):
    edited_image: str | None = None
    error: str | None = None

try:
    gcp_project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    gcp_location = "global"
    
    if not gcp_project_id:
        raise ValueError("GCP_PROJECT_ID environment variable not set.")
    
    client = genai.Client(
        vertexai=True,
        project=gcp_project_id,
        location=gcp_location,
        http_options=HttpOptions(api_version="v1")
    )
except Exception as e:
    print(f"Error initializing Gemini Client: {e}")

def edit_image(image_bytes: bytes, prompt: str) -> EditImageResponse:
    """
    Edits an image based on a text prompt using the Gemini API.

    Args:
        image_bytes (bytes): Raw bytes of the input image.
        prompt (str): Text prompt describing the edits.

    Returns:
        EditImageResponse: Pydantic model instance with the result.
    """
    try:
        msg_image = Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        msg_prompt = Part.from_text(text=prompt)

        model = "gemini-2.5-flash-image-preview"
        contents = [
            Content(role="user", parts=[msg_image, msg_prompt])
        ]

        generate_content_config = GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=32768,
            response_modalities=["TEXT", "IMAGE"],
            safety_settings=[
                SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                SafetySetting(category="HARM_CATEGORY_IMAGE_HATE", threshold="OFF"),
                SafetySetting(category="HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT", threshold="OFF"),
                SafetySetting(category="HARM_CATEGORY_IMAGE_HARASSMENT", threshold="OFF"),
                SafetySetting(category="HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT", threshold="OFF")
            ]
        )

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config
        )

        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.inline_data and part.inline_data.mime_type == 'image/png':
                    output_image_data = part.inline_data.data
                    base64_image = base64.b64encode(output_image_data).decode('utf-8')
                    return EditImageResponse(edited_image=base64_image)

        return EditImageResponse(error="No edited image generated.")
    
    except Exception as e:
        return EditImageResponse(error=str(e))
