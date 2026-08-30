"""Docling-backed document conversion mapped to EEGAgent structure blocks."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


SUPPORTED_DOCUMENT_EXTENSIONS = frozenset(
    {".pdf", ".docx", ".md", ".markdown", ".txt", ".html", ".xhtml"}
)

_SKIPPED_LABELS = {"page_header", "page_footer"}
_HEADING_LABELS = {"title", "section_header"}


def _label_name(item: Any) -> str:
    label = getattr(item, "label", "")
    value = getattr(label, "value", None)
    if value is None:
        value = getattr(label, "name", label)
    return str(value).strip().lower()


def _page_number(item: Any) -> int | None:
    provenance = getattr(item, "prov", None) or []
    if not provenance:
        return None
    page = getattr(provenance[0], "page_no", None)
    return int(page) if page is not None else None


def _item_text(item: Any, document: Any) -> str:
    text = getattr(item, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    exporter = getattr(item, "export_to_markdown", None)
    if callable(exporter):
        try:
            return str(exporter(doc=document)).strip()
        except TypeError:
            return str(exporter()).strip()
    return ""


def _heading_level(item: Any, traversal_level: int, label: str) -> int:
    if label == "title":
        return 0
    declared = getattr(item, "level", None)
    if isinstance(declared, int) and declared > 0:
        return min(declared, 6)
    return min(max(int(traversal_level), 1), 6)


def docling_document_to_blocks(document: Any) -> list[dict[str, Any]]:
    """Map a DoclingDocument into the block schema consumed by EEGAgent chunking."""
    blocks: list[dict[str, Any]] = []
    heading_stack: dict[int, str] = {}

    for item, traversal_level in document.iterate_items():
        label = _label_name(item)
        if label in _SKIPPED_LABELS:
            continue
        text = _item_text(item, document)
        if not text:
            continue
        page = _page_number(item)

        if label in _HEADING_LABELS:
            level = _heading_level(item, traversal_level, label)
            heading_stack[level] = text
            for old_level in [value for value in heading_stack if value > level]:
                del heading_stack[old_level]
            title_path = [heading_stack[value] for value in sorted(heading_stack)]
            blocks.append(
                {
                    "type": "heading",
                    "text": text,
                    "page": page,
                    "heading_level": max(level, 1),
                    "title_path": title_path,
                }
            )
            continue

        block_type = "paragraph"
        if label == "list_item":
            block_type = "list"
        elif label == "table":
            block_type = "table"
        blocks.append(
            {
                "type": block_type,
                "text": text,
                "page": page,
                "title_path": [heading_stack[value] for value in sorted(heading_stack)],
            }
        )
    return blocks


@lru_cache(maxsize=1)
def _document_converter():
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import HeadingHierarchyOptions, PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Docling is required to ingest RAG documents. Install the project requirements."
        ) from exc

    pdf_options = PdfPipelineOptions()
    pdf_options.do_ocr = True
    pdf_options.do_table_structure = True
    pdf_options.generate_parsed_pages = True
    pdf_options.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)
    pdf_options.document_timeout = 120
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.MD, InputFormat.HTML],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)},
    )


def parse_document(filepath: str, *, converter=None) -> list[dict[str, Any]]:
    """Convert one supported source with Docling and return EEGAgent blocks."""
    path = Path(filepath).resolve()
    if path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
        raise ValueError(f"Unsupported knowledge document format: {path.suffix}. Expected: {supported}")
    if not path.is_file():
        raise FileNotFoundError(f"Knowledge document not found: {path}")
    result = (converter or _document_converter()).convert(path)
    blocks = docling_document_to_blocks(result.document)
    if not blocks:
        raise ValueError(f"Docling produced no structured content for: {path}")
    return blocks
