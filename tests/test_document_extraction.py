"""Tests for PdfTextExtractor. Uses reportlab (dev-only dependency) to
generate real PDF fixtures in-memory, rather than fixture files."""

from io import BytesIO

from reportlab.pdfgen import canvas

from agent_server.document_extraction import PdfTextExtractor


def _make_pdf(pages_text: list[str]) -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    for text in pages_text:
        pdf.drawString(72, 720, text)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def test_extracts_text_from_single_page_pdf():
    pdf_bytes = _make_pdf(["Hotel Invoice: EUR 120.00"])

    result = PdfTextExtractor().extract(pdf_bytes)

    assert "Hotel Invoice: EUR 120.00" in result.text
    assert result.page_count == 1
    assert result.pages_with_no_text == []
    assert not result.is_likely_scanned


def test_extracts_text_from_multi_page_pdf_in_order():
    pdf_bytes = _make_pdf(["Page one content", "Page two content"])

    result = PdfTextExtractor().extract(pdf_bytes)

    assert result.page_count == 2
    assert result.text.index("Page one content") < result.text.index("Page two content")


def test_scanned_pdf_detected_via_empty_text_pages():
    # A page with no text-drawing calls at all simulates a scanned/image-only
    # page - no text layer to extract.
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.showPage()
    pdf.save()

    result = PdfTextExtractor().extract(buffer.getvalue())

    assert result.page_count == 1
    assert result.pages_with_no_text == [0]
    assert result.is_likely_scanned
    assert result.text.strip() == ""
