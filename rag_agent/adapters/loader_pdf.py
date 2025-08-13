from __future__ import annotations
from typing import List
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from pdf2image import convert_from_path
import pytesseract, os
from rag_agent.core.dto    import DocumentChunk
from rag_agent.core.errors import LoaderError
from rag_agent.settings    import settings
from . import register_loader

# Make sure TESSERACT_CMD is configured in your .env / settings
TESS_CMD = os.getenv("TESSERACT_CMD") or settings.tesseract_cmd
if TESS_CMD:  
    pytesseract.pytesseract.tesseract_cmd = TESS_CMD

@register_loader("pdf")
class PDFLoader:
    def load(self, file_path: str) -> List[DocumentChunk]:
        try:
            # fast text-based load
            docs = PyPDFLoader(file_path).load()
            total = sum(len(d.page_content) for d in docs)
            if total < settings.pdf_ocr_threshold:
                raise ValueError("too few chars, fallback to OCR")
        except Exception:
            # OCR fallback
            pages = convert_from_path(
                file_path,
                dpi=settings.ocr_dpi,
                poppler_path=settings.poppler_path,
            )
            docs = []
            for i, img in enumerate(pages):
                text = pytesseract.image_to_string(img)
                if text.strip():
                    docs.append(DocumentChunk(
                        id=f"{file_path}-ocr-{i}",
                        text=text,
                        metadata={"source": file_path, "page": i, "ocr": True},
                    ))
        else:
            # wrap langchain docs into DocumentChunk
            docs = [
                DocumentChunk(
                    id=f"{file_path}-{i}",
                    text=d.page_content,
                    metadata={"source": file_path, "page": getattr(d, "page_number", i)},
                )
                for i, d in enumerate(docs)
            ]

        # simple chunking to settings.chunk_size words
        chunks: list[DocumentChunk] = []
        buf, idx = "", 0
        for dc in docs:
            buf += dc.text + "\n"
            if len(buf.split()) >= settings.chunk_size:
                chunks.append(DocumentChunk(
                    id=f"{file_path}-chunk-{idx}",
                    text=buf.strip(),
                    metadata=dc.metadata,
                ))
                idx += 1
                buf = ""
        if buf.strip():
            chunks.append(DocumentChunk(
                id=f"{file_path}-chunk-{idx}",
                text=buf.strip(),
                metadata=docs[-1].metadata,
            ))

        return chunks
