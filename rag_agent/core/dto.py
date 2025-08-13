from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class QuestionDTO(BaseModel):
    text: str
    user_id: str | None = None

class Claim(BaseModel):
    id: str
    text: str
    citation_ids: List[str] = Field(default_factory=list)
    confidence: float | None = None
    support_citation_id: Optional[str] = None
    verified: Optional[bool] = None

class Citation(BaseModel):
    chunk_id: str
    page: int | None = None
    snippet: str

class AnswerDTO(BaseModel):
    text: str
    citations: List[Citation] = Field(default_factory=list)
    claims: List[Claim] = Field(default_factory=list)

class DocumentChunk(BaseModel):
    id: str
    text: str
    metadata: Dict = Field(default_factory=dict)
