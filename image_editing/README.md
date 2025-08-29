Image Editing Module
====================

Overview
--------

This module provides an image editing tool powered by the **Google Gemini API**. It allows users to upload an image and provide a **text prompt** to edit the image (e.g., add elements, change styles, or modify content). The module processes the input using Gemini's multimodal capabilities and returns the edited image in **Base64-encoded PNG** format.

Endpoints Added
---------------

*   **/api/demo\_backend\_v2/image\_editing/edit** (POST): Edits the uploaded image based on the text prompt and returns the edited image in **PNG** format.
    
    *   **Request**: Multipart form-data with:
        
        *   **image** (file): The image file to be edited.
            
        *   **prompt** (text): The text description of the edits to be applied.
            
    *   **Response**: A **streaming response** with the edited image in **PNG** format.
        

### Example Request using curl:
`curl -X 'POST' \'http://127.0.0.1:8000/api/demo_backend_v2/image_editing/edit' \-F 'image=@/path/to/input_image.png' \-F 'prompt=Add a red hat to the person in the image'`

### Example Response (success):

`HTTP/1.1 200 OK  Content-Type: image/png  Content-Disposition: inline; filename="edited_image.png"`

You will receive the **image** as a direct download or display in your browser.

### Example Response (error):
`{"error": "Error message here"}`

Setup/Installation Steps
------------------------

1.  Install the dependencies with: `pip install -r requirements.txt`
    
    *   Ensure the required Python packages are installed by adding the dependencies to your requirements.txt file:
        
2.  **Set up Google Cloud project**:
    
    *   Example: export GOOGLE\_CLOUD\_PROJECT="your\_project\_id"
        
3.  The FastAPI server will start running on http://127.0.0.1:8000.
    
    *   uvicorn app:app --reload
        

How It Works
------------

1.  **User uploads an image** and provides a **text prompt** describing the desired edits (e.g., "Add a red hat to the person in the image").
    
2.  The image and prompt are sent to the **Google Gemini API** for processing.
    
3.  The **Gemini API** edits the image according to the prompt and returns the edited image in **Base64-encoded PNG format**.
    
4.  The **FastAPI server** decodes the Base64 string and returns the **image as PNG** in the response.
    

Example Request/Response Flow
-----------------------------

*   **Request**: The user uploads an image and provides a prompt describing the changes.
    
*   **Response**: The server processes the image and returns the **edited image** in PNG format.
    

Dependencies Used
-----------------

*   **google-generativeai**: For interacting with the **Google Gemini API** to process the image and prompt.
    
*   **pillow**: For handling image processing, including converting the uploaded image to the necessary format for the API.
    
*   **fastapi**: The framework used to create the API endpoint.
    
*   **uvicorn**: ASGI server to run the FastAPI application.
    
*   **pydantic**: For request and response validation.
    

Notes
-----

*   The **Google Gemini API** requires **authentication** using your **Google Cloud credentials**. Ensure your environment is properly set up with the correct **API keys** or **service account credentials**.
    
*   Make sure to monitor the quota and usage limits associated with your Google Cloud project to avoid hitting any API limits.  