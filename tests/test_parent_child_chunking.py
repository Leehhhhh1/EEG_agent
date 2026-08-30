import unittest

from RAG.chunker import (
    _block_units,
    build_child_chunks,
    build_parent_chunks,
    expand_child_window,
    merge_child_texts,
)


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


class ParentChildChunkingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = WordTokenizer()

    def test_builds_section_parent_and_linked_children(self):
        blocks = [
            {"type": "heading", "text": "LPD", "page": 1, "title_path": ["LPD"]},
            {
                "type": "paragraph",
                "text": "One two three four. Five six seven eight. Nine ten eleven twelve.",
                "page": 1,
                "title_path": ["LPD"],
            },
        ]
        parents = build_parent_chunks(blocks, "guide.pdf", tokenizer=self.tokenizer)
        children = build_child_chunks(
            parents[0],
            tokenizer=self.tokenizer,
            target_tokens=4,
            max_tokens=6,
            overlap_tokens=2,
        )
        self.assertEqual(len(parents), 1)
        self.assertGreaterEqual(len(children), 2)
        self.assertEqual(len({child["text"] for child in children}), len(children))
        self.assertTrue(all(child["token_count"] <= 6 for child in children))
        self.assertTrue(all(child["parent_id"] == parents[0]["parent_id"] for child in children))
        self.assertTrue(all(child["retrieval_text"].startswith("LPD\n") for child in children))

    def test_expands_match_with_adjacent_children(self):
        children = [
            {"chunk_id": f"p:child:{index}", "parent_id": "p", "child_index": index, "text": text}
            for index, text in enumerate(["previous", "matched", "next", "far"])
        ]
        context = expand_child_window(children[1], children, window=1)
        self.assertEqual(context, "previous\n\nmatched\n\nnext")
        self.assertNotIn("far", context)

    def test_neighbor_context_removes_overlap_duplicates(self):
        context = merge_child_texts([
            {"text": "First sentence. Shared sentence."},
            {"text": "Shared sentence. Last sentence."},
        ])
        self.assertEqual(context.count("Shared sentence."), 1)

    def test_sentence_boundaries_support_chinese_without_spaces(self):
        units = _block_units(
            {"type": "paragraph", "text": "第一句。第二句！第三句？"},
            max_tokens=20,
            tokenizer=self.tokenizer,
        )
        self.assertEqual(units, ["第一句。", "第二句！", "第三句？"])

    def test_english_sentence_boundaries_still_require_whitespace(self):
        units = _block_units(
            {"type": "paragraph", "text": "First sentence. Second sentence!Third fragment."},
            max_tokens=20,
            tokenizer=self.tokenizer,
        )
        self.assertEqual(units, ["First sentence.", "Second sentence!Third fragment."])


if __name__ == "__main__":
    unittest.main()
