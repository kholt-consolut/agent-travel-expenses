"""PDF text extraction.

First building block of the file-upload feature: the agent should be able
to read an uploaded PDF's content. Kept separate from upload/staging
concerns (which come later) so each piece can be extended independently -
e.g. an OCR fallback for scanned PDFs, or other file types, without
touching this class.
"""

from dataclasses import dataclass, field
from io import BytesIO

from pypdf import PdfReader


@dataclass
class PdfExtractionResult:
    text: str
    page_count: int
    pages_with_no_text: list[int] = field(default_factory=list)

    @property
    def is_likely_scanned(self) -> bool:
        """True if every page came back with no extractable text layer -
        the PDF is probably a scan/image and needs OCR, which this class
        does not do."""
        return self.page_count > 0 and len(self.pages_with_no_text) == self.page_count


class PdfTextExtractor:
    """Extracts text from a PDF's text layer.

    Does not do OCR: a scanned/image-only PDF comes back with empty text
    per page (see `PdfExtractionResult.is_likely_scanned`) rather than
    raising, since that's a legitimate outcome to detect and handle, not an
    error.
    """

    def extract(self, pdf_bytes: bytes) -> PdfExtractionResult:
        reader = PdfReader(BytesIO(pdf_bytes))
        page_texts: list[str] = []
        pages_with_no_text: list[int] = []

        for index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if not text.strip():
                pages_with_no_text.append(index)
            page_texts.append(text)

        return PdfExtractionResult(
            text="\n\n".join(page_texts),
            page_count=len(reader.pages),
            pages_with_no_text=pages_with_no_text,
        )
