import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from RAG.docling_parser import docling_document_to_blocks, parse_document


class FakeItem:
    def __init__(self, label, text=None, *, level=None, page=None, markdown=None):
        self.label = SimpleNamespace(value=label)
        self.text = text
        self.level = level
        self.prov = [] if page is None else [SimpleNamespace(page_no=page)]
        self._markdown = markdown

    def export_to_markdown(self, doc=None):
        return self._markdown or ""


class FakeDocument:
    def __init__(self, items):
        self.items = items

    def iterate_items(self):
        return iter(self.items)


class DoclingParserTests(unittest.TestCase):
    def test_maps_docling_structure_to_eegagent_blocks(self):
        document = FakeDocument([
            (FakeItem("title", "EEG Guide", page=1), 1),
            (FakeItem("page_header", "Repeated header", page=1), 1),
            (FakeItem("section_header", "Findings", level=1, page=2), 1),
            (FakeItem("paragraph", "Background is described here.", page=2), 2),
            (FakeItem("section_header", "Terminology", level=2, page=3), 2),
            (FakeItem("list_item", "Periodic discharge", page=3), 3),
            (FakeItem("table", page=4, markdown="| Term | Meaning |"), 3),
            (FakeItem("page_footer", "4", page=4), 1),
        ])

        blocks = docling_document_to_blocks(document)

        self.assertEqual([block["type"] for block in blocks], [
            "heading", "heading", "paragraph", "heading", "list", "table"
        ])
        self.assertEqual(blocks[2]["title_path"], ["EEG Guide", "Findings"])
        self.assertEqual(
            blocks[4]["title_path"], ["EEG Guide", "Findings", "Terminology"]
        )
        self.assertEqual(blocks[5]["text"], "| Term | Meaning |")
        self.assertEqual(blocks[5]["page"], 4)

    def test_parse_document_uses_injected_docling_converter(self):
        document = FakeDocument([
            (FakeItem("section_header", "Section", level=1), 1),
            (FakeItem("text", "Content"), 2),
        ])
        converter = SimpleNamespace(
            convert=lambda path: SimpleNamespace(document=document)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guide.md"
            path.write_text("# ignored by fake converter", encoding="utf-8")
            blocks = parse_document(str(path), converter=converter)

        self.assertEqual(blocks[-1]["text"], "Content")
        self.assertEqual(blocks[-1]["title_path"], ["Section"])

    def test_rejects_unsupported_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guide.xyz"
            path.write_text("content", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported knowledge document format"):
                parse_document(str(path), converter=SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
