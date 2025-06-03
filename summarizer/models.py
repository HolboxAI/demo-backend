from pydantic import BaseModel

class UploadResponse(BaseModel):
    pdf_id: str
    message: str

class SummaryRequest(BaseModel):
    pdf_id: str

class SummaryResponse(BaseModel):
    summary: str
