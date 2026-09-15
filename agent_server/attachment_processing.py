"""Wires PDF attachment handling into the request pipeline: extracts text
for the agent to read, and stages the file to SAP - the two concerns from
document_extraction.py and sap_file_upload.py.

Only PDF `input_file` content parts (OpenAI Responses API input shape,
https://platform.openai.com/docs/api-reference/responses/create) are
handled; anything else (other file types, or `file_id`/`file_url` instead
of `file_data`) is left untouched for now.
"""

import asyncio
import base64

from agent_server.document_extraction import PdfExtractionResult, PdfTextExtractor
from agent_server.sap_file_upload import SapFileUploader


def _decode_file_data(file_data: str) -> bytes:
    """`file_data` is a data URI (`data:<mime>;base64,<data>`) per the
    OpenAI Responses API spec; fall back to treating it as raw base64."""
    if file_data.startswith("data:") and ";base64," in file_data:
        file_data = file_data.split(";base64,", 1)[1]
    return base64.b64decode(file_data)


def _is_pdf_part(part: dict) -> bool:
    if part.get("type") != "input_file" or not part.get("file_data"):
        return False
    return part.get("filename", "").lower().endswith(".pdf")


def _extraction_to_text(filename: str, result: PdfExtractionResult) -> str:
    if result.is_likely_scanned:
        return f"[Attached PDF '{filename}': no extractable text - likely a scanned document]"
    return f"[Attached PDF '{filename}']\n{result.text}"


class AttachmentProcessor:
    """Replaces PDF `input_file` content parts with their extracted text,
    and stages each file to SAP - concurrently, per attachment."""

    def __init__(self, pdf_extractor: PdfTextExtractor, uploader: SapFileUploader):
        self._pdf_extractor = pdf_extractor
        self._uploader = uploader

    async def process(self, messages: list[dict]) -> list[dict]:
        pdf_parts = [
            part
            for message in messages
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if isinstance(part, dict) and _is_pdf_part(part)
        ]
        if not pdf_parts:
            return messages

        replacements = await asyncio.gather(*(self._process_one(part) for part in pdf_parts))
        replacement_by_id = {id(part): replacement for part, replacement in zip(pdf_parts, replacements)}

        return [
            {**message, "content": [replacement_by_id.get(id(part), part) for part in message["content"]]}
            if isinstance(message.get("content"), list)
            else message
            for message in messages
        ]

    async def _process_one(self, part: dict) -> dict:
        filename = part["filename"]
        content = _decode_file_data(part["file_data"])

        extraction, _ = await asyncio.gather(
            asyncio.to_thread(self._pdf_extractor.extract, content),
            self._uploader.upload(filename, content),
        )
        return {"type": "input_text", "text": _extraction_to_text(filename, extraction)}
