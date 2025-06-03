from pydantic import BaseModel

class UploadResponse(BaseModel):
    pdf_id: str
    message: str

class QuestionRequestPDF(BaseModel):
    pdf_id: str
    question: str

class AnswerResponse(BaseModel):
    answer: str
    source_chunks: list[str]
