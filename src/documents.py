"""Bounded, non-executing extraction with explicit page/section provenance."""

import csv
from dataclasses import dataclass
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import re
import unicodedata
from xml.etree import ElementTree as ET
from zipfile import ZipFile, BadZipFile

from src.config import Settings

SUPPORTED = {".pdf", ".docx", ".txt", ".csv", ".json"}


class ExtractionError(ValueError):
    pass


@dataclass(frozen=True)
class Section:
    text: str
    page_number: int | None = None
    heading: str = ""
    kind: str = "text"
    extraction: str = "native"


def clean_text(text):
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    text = re.sub(r"[^\S\n]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def validate_file(path, settings: Settings):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ExtractionError("Supply a regular file, not a symlink")
    if path.suffix.lower() not in SUPPORTED:
        raise ExtractionError("Supported formats: PDF, DOCX, TXT, CSV, JSON")
    if not 0 < path.stat().st_size <= settings.max_upload_bytes:
        raise ExtractionError("File is empty or exceeds the configured size limit")
    return path


def _text_sections(text, page=None, method="native"):
    heading, pending = "", []
    for line in clean_text(text).splitlines():
        stripped = line.strip()
        is_heading = (stripped.startswith("#") or
                      (len(stripped) < 90 and stripped and
                       (stripped.isupper() or re.match(r"^\d+(?:\.\d+)*[.)]?\s+[A-Z]", stripped))))
        if is_heading:
            if pending:
                yield Section("\n".join(pending), page, heading, extraction=method)
                pending = []
            heading = stripped.lstrip("# ")
        else:
            pending.append(line)
    if any(s.strip() for s in pending):
        yield Section(clean_text("\n".join(pending)), page, heading, extraction=method)
    elif heading:
        yield Section(heading, page, heading, extraction=method)


def _safe_docx(path, settings):
    try:
        with ZipFile(path) as archive:
            entries = archive.infolist()
            if (len(entries) > 2000 or sum(e.file_size for e in entries) > settings.max_upload_bytes * 10
                    or any(e.flag_bits & 1 for e in entries)):
                raise ExtractionError("DOCX archive is encrypted or exceeds expansion limits")
            if archive.getinfo("word/document.xml").file_size > settings.max_text_chars * 10:
                raise ExtractionError("DOCX XML exceeds the text expansion limit")
            data = archive.read("word/document.xml")
            if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
                raise ExtractionError("XML entities are not allowed")
            root = ET.fromstring(data)
    except (BadZipFile, KeyError, ET.ParseError) as exc:
        raise ExtractionError("Invalid DOCX document") from exc
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    body = root.find("w:body", ns)
    if body is None:
        raise ExtractionError("DOCX body missing")
    heading = ""
    for node in body:
        if node.tag.endswith("}p"):
            text = clean_text("".join(n.text or "" for n in node.findall(".//w:t", ns)))
            style = node.find("w:pPr/w:pStyle", ns)
            if style is not None and "heading" in next(iter(style.attrib.values()), "").lower():
                heading = text
            elif text:
                yield Section(text, heading=heading)
        elif node.tag.endswith("}tbl"):
            rows = []
            for row in node.findall("w:tr", ns):
                rows.append(" | ".join(clean_text(" ".join(n.text or "" for n in cell.findall(".//w:t", ns)))
                                       for cell in row.findall("w:tc", ns)))
            yield Section("\n".join(rows), heading=heading, kind="table")
    # DOCX has no stable page layout: never manufacture page numbers.


def _faq_sections(rows):
    if not isinstance(rows, list) or not rows:
        raise ExtractionError("Structured FAQ must be a nonempty list")
    for row in rows:
        if not isinstance(row, dict):
            raise ExtractionError("FAQ rows must be objects")
        # Explicitly reject benchmark data even if it also has question/answer.
        if any(k in row for k in ("expected_answer", "expected_chunk_ids", "answerability", "split", "difficulty")):
            raise ExtractionError("Evaluation data cannot enter production ingestion")
        if row.get("status") != "PUBLISHED" or row.get("verified") not in (True, "true", "True"):
            raise ExtractionError("FAQ imports require verified=true and status=PUBLISHED")
        question, answer = row.get("question"), row.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str) or not question.strip() or not answer.strip():
            raise ExtractionError("FAQ question and answer must be nonempty strings")
        yield Section(f"Question: {clean_text(question)}\nAnswer: {clean_text(answer)}",
                      heading=str(row.get("category", "FAQ")), kind="faq")


def extract(path, settings=None, ocr=None):
    settings = settings or Settings.from_env()
    path = validate_file(path, settings)
    suffix = path.suffix.lower()
    sections = []
    try:
        if suffix == ".pdf":
            if not path.read_bytes().startswith(b"%PDF-"):
                raise ExtractionError("Invalid PDF signature")
            from pypdf import PdfReader
            reader = PdfReader(path)
            if reader.is_encrypted:
                raise ExtractionError("Encrypted PDFs are not supported")
            if len(reader.pages) > settings.max_pages:
                raise ExtractionError("PDF exceeds configured page limit")
            for number, page in enumerate(reader.pages, 1):
                text = clean_text(page.extract_text() or "")
                method = "native"
                if len(text) < 20:
                    if ocr is None:
                        raise ExtractionError(f"Page {number} needs OCR; enable the OCR extra and Tesseract")
                    text = clean_text(ocr(path, number))
                    method = "ocr"
                    if len(text) < 20:
                        raise ExtractionError(f"OCR produced insufficient text on page {number}")
                sections.extend(_text_sections(text, number, method))
            # Optional table extraction adds structured rows, retaining page metadata.
            try:
                import pdfplumber
            except ImportError:
                pass
            else:
                with pdfplumber.open(path) as pdf:
                    for number, page in enumerate(pdf.pages, 1):
                        for table in page.extract_tables():
                            text = "\n".join(" | ".join(clean_text(str(v or "")) for v in row) for row in table)
                            if text.strip():
                                sections.append(Section(text, number, kind="table"))
        elif suffix == ".docx":
            sections = list(_safe_docx(path, settings))
        elif suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(payload, dict):
                if set(payload) != {"faqs"}:
                    raise ExtractionError("JSON must contain only a faqs array; evaluation exports are rejected")
                payload = payload["faqs"]
            sections = list(_faq_sections(payload))
        elif suffix == ".csv":
            rows = list(csv.DictReader(StringIO(path.read_text(encoding="utf-8-sig"))))
            if rows and "question" in rows[0]:
                sections = list(_faq_sections(rows))
            elif rows:
                if any(k in rows[0] for k in ("expected_answer", "split", "answerability")):
                    raise ExtractionError("Evaluation CSV cannot enter production ingestion")
                sections = [Section(" | ".join(rows[0]) + "\n" + "\n".join(" | ".join(r.values()) for r in rows), kind="table")]
        else:
            sections = list(_text_sections(path.read_text(encoding="utf-8-sig")))
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"Could not extract {suffix} document ({type(exc).__name__})") from exc
    sections = [s for s in sections if s.text.strip()]
    if not sections or sum(len(s.text) for s in sections) > settings.max_text_chars:
        raise ExtractionError("Document is empty or exceeds the extracted-text limit")
    return sections


def tesseract_ocr(path, page_number):
    """Explicit opt-in; fixed local OCR executable, never source-supplied commands."""
    import fitz
    import pytesseract
    from PIL import Image
    with fitz.open(path) as document:
        page = document[page_number - 1]
        if page.rect.width * page.rect.height * 4 > 30_000_000:
            raise ExtractionError("OCR page exceeds image-size limit")
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csRGB)
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        return pytesseract.image_to_string(image, timeout=45)


def content_hash(sections):
    canonical = "\n".join(clean_text(s.text) for s in sections)
    return sha256(canonical.encode("utf-8")).hexdigest()
