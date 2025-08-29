# Image Editing Module

## Overview
This module provides an image editing tool powered by the Google Gemini API. It allows users to upload an image and provide a text prompt to edit the image (e.g., add elements, change styles, or modify content). The module processes the input using Gemini's multimodal capabilities and returns the edited image as a base64-encoded PNG string.

## Endpoints Added
- **/image_editing/edit** (POST): Edits the uploaded image based on the text prompt.
  - Request: Multipart form-data with 'image' (file) and 'prompt' (text).
  - Response: JSON object with 'edited_image' (base64 string) or 'error' (string).

## Setup/Installation Steps
1. Ensure the Google Generative AI SDK is installed (added to requirements.txt).
2. Set the GEMINI_API_KEY environment variable with your Google Gemini API key. You can obtain this from Google AI Studio.
   - Example: export GEMINI_API_KEY=your_api_key_here
3. No additional setup required beyond Flask app integration.

## Example Request/Response
### Request (using curl):
 
curl -X POST http://localhost:5000/image_editing/edit -F 'image=@/path/to/input_image.png' -F 'prompt=Add a red hat to the person in the image'
 
### Response (success):
 
{ "edited_image": "iVBORw0KGgoAAAANSUhEUgAA... (base64-encoded PNG data)" }
 
### Response (error):
  
{ "error": "Error message here" }

## Dependencies Used
- google-generativeai: For interacting with the Gemini API.
- pillow: For handling image processing and opening uploaded files.