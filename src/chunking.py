"""Deterministic, bounded section chunks with complete source metadata.

UTF-8 bytes are a conservative token upper bound (not a claim to be Qwen tokens).
This offline default cannot silently overflow a subword tokenizer. A real model
tokenizer can be supplied for experiments. Embedding windows are handled separately.
"""

from dataclasses import dataclass
from hashlib import sha256
import re


def token_count(text):
    return len(text.encode("utf-8"))


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict


def split_text(text, size, overlap, count=token_count):
    # Split at paragraph/sentence/word boundaries when possible. Character
    # fallback guarantees progress even on languages without spaces.
    start = 0
    while start < len(text):
        low, high = start + 1, len(text)
        while low <= high:
            middle = (low + high) // 2
            if count(text[start:middle]) <= size:
                low = middle + 1
            else:
                high = middle - 1
        end = high
        if end <= start:
            raise ValueError("Token budget cannot fit a single character")
        if end < len(text):
            matches = list(re.finditer(r"\n\n|(?<=[.!?])\s+|\n|\s+", text[start:end]))
            eligible = [m.end() for m in matches if m.end() >= (end - start) * .6]
            if eligible:
                end = start + eligible[-1]
        piece = text[start:end].strip()
        if piece:
            yield piece
        if end == len(text):
            break
        next_start = end
        while next_start > start + 1 and count(text[next_start - 1:end]) <= overlap:
            next_start -= 1
        start = max(start + 1, next_start)


def chunk_sections(sections, document_id, version_id, source, size=500, overlap=75, count=token_count):
    if not 0 <= overlap < size:
        raise ValueError("Overlap must be less than chunk size")
    chunks = []
    for section_index, section in enumerate(sections):
        parent = sha256(f"{version_id}:{section_index}".encode()).hexdigest()[:24]
        # Keep an FAQ pair or a table intact if it fits; oversized tables split
        # along row/word boundaries and retain their section parent and heading.
        if count(section.text) <= size:
            pieces = [section.text]
        elif section.kind == "table" and "\n" in section.text:
            header, body = section.text.split("\n", 1)
            remaining = size - count(header + "\n")
            if remaining > max(20, overlap):
                pieces = [header + "\n" + piece for piece in split_text(body, remaining, min(overlap, remaining // 5), count)]
            else:
                pieces = list(split_text(section.text, size, overlap, count))
        else:
            pieces = list(split_text(section.text, size, overlap, count))
        for child, text in enumerate(pieces):
            identity = f"{version_id}:{parent}:{child}:{size}:{overlap}:{text}"
            chunk_id = "CHK_" + sha256(identity.encode()).hexdigest()[:28]
            metadata = {"document_id": document_id, "version_id": version_id,
                        "chunk_id": chunk_id, "source": source,
                        "page_number": section.page_number, "heading": section.heading,
                        "kind": section.kind, "extraction": section.extraction,
                        "parent_id": parent, "child_index": child,
                        "token_count": count(text)}
            chunks.append(Chunk(chunk_id, text, metadata))
    return chunks
