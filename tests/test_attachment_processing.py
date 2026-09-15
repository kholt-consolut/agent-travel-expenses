"""Tests for AttachmentProcessor. Uses fake PdfTextExtractor/SapFileUploader
doubles (not real PDFs or network) to isolate the wiring logic: which parts
get replaced, that upload and extraction both run, and that non-PDF/
non-attachment content is left untouched."""

import asyncio
import base64

from agent_server.attachment_processing import AttachmentProcessor
from agent_server.document_extraction import PdfExtractionResult


class _FakeExtractor:
    def __init__(self, result: PdfExtractionResult):
        self._result = result
        self.calls: list[bytes] = []

    def extract(self, pdf_bytes: bytes) -> PdfExtractionResult:
        self.calls.append(pdf_bytes)
        return self._result


class _FakeUploader:
    def __init__(self):
        self.calls: list[tuple[str, bytes]] = []

    async def upload(self, file_name: str, content: bytes):
        self.calls.append((file_name, content))
        return {"status": "ok"}


def _data_uri(raw: bytes) -> str:
    return "data:application/pdf;base64," + base64.b64encode(raw).decode("ascii")


def test_replaces_pdf_input_file_with_extracted_text_and_uploads():
    extractor = _FakeExtractor(PdfExtractionResult(text="Hotel EUR 120", page_count=1))
    uploader = _FakeUploader()
    processor = AttachmentProcessor(extractor, uploader)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Here's my receipt"},
                {"type": "input_file", "filename": "receipt.pdf", "file_data": _data_uri(b"pdf-bytes")},
            ],
        }
    ]

    result = asyncio.run(processor.process(messages))

    content = result[0]["content"]
    assert content[0] == {"type": "input_text", "text": "Here's my receipt"}
    assert content[1]["type"] == "input_text"
    assert "Hotel EUR 120" in content[1]["text"]
    assert "receipt.pdf" in content[1]["text"]

    assert extractor.calls == [b"pdf-bytes"]
    assert uploader.calls == [("receipt.pdf", b"pdf-bytes")]


def test_scanned_pdf_gets_placeholder_text_not_empty_string():
    extractor = _FakeExtractor(PdfExtractionResult(text="", page_count=1, pages_with_no_text=[0]))
    uploader = _FakeUploader()
    processor = AttachmentProcessor(extractor, uploader)

    messages = [
        {
            "role": "user",
            "content": [{"type": "input_file", "filename": "scan.pdf", "file_data": _data_uri(b"x")}],
        }
    ]

    result = asyncio.run(processor.process(messages))

    text = result[0]["content"][0]["text"]
    assert "no extractable text" in text
    assert "scan.pdf" in text


def test_non_pdf_attachments_and_plain_messages_are_left_untouched():
    extractor = _FakeExtractor(PdfExtractionResult(text="unused", page_count=1))
    uploader = _FakeUploader()
    processor = AttachmentProcessor(extractor, uploader)

    messages = [
        {"role": "user", "content": "plain string content"},
        {
            "role": "user",
            "content": [{"type": "input_file", "filename": "photo.png", "file_data": "data:image/png;base64,abc"}],
        },
    ]

    result = asyncio.run(processor.process(messages))

    assert result == messages
    assert extractor.calls == []
    assert uploader.calls == []


def test_multiple_pdfs_are_processed_concurrently_and_independently():
    extractor = _FakeExtractor(PdfExtractionResult(text="shared text", page_count=1))
    uploader = _FakeUploader()
    processor = AttachmentProcessor(extractor, uploader)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_file", "filename": "a.pdf", "file_data": _data_uri(b"aaa")},
                {"type": "input_file", "filename": "b.pdf", "file_data": _data_uri(b"bbb")},
            ],
        }
    ]

    result = asyncio.run(processor.process(messages))

    filenames_uploaded = {name for name, _ in uploader.calls}
    assert filenames_uploaded == {"a.pdf", "b.pdf"}
    assert len(result[0]["content"]) == 2
    assert all(part["type"] == "input_text" for part in result[0]["content"])
