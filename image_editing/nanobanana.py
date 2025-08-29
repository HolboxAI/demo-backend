import os
import base64
import io
from PIL import Image
import google.generativeai as genai
from pydantic import BaseModel

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

class EditImageRequest(BaseModel):
    image_data: str
    prompt: str

class EditImageResponse(BaseModel):
    edited_image: str | None = None
    error: str | None = None

def edit_image(image_data: str, prompt: str):
    """
    Edits an image based on a text prompt using the Gemini API.
    
    Args:
        image_data (str): Base64-encoded string of the input image.
        prompt (str): Text prompt describing the edits.
    
    Returns:
        EditImageResponse: Pydantic model instance with the result.
    """
    try:        
        # Decode the base64 image data
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        response = model.generate_content(
            [prompt, img],
            generation_config=genai.types.GenerationConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if 'inline_data' in part and part.inline_data.mime_type == 'image/png':
                    output_image_data = part.inline_data.data  # bytes
                    base64_image = base64.b64encode(output_image_data).decode('utf-8')
                    return EditImageResponse(edited_image=base64_image)
        
        return EditImageResponse(error='No edited image generated.')
    
    except Exception as e:
        return EditImageResponse(error=str(e))
