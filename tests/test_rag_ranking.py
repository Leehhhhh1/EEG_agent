import unittest

from RAG.ranking import fuse_scores_and_deduplicate_sources


class RAGRankingTests(unittest.TestCase):
    def test_uses_point_two_point_eight_weighting(self):
        result = fuse_scores_and_deduplicate_sources(
            [{"text": "a", "source": "a.pdf", "coarse_score": 0.5}],
            [0.75],
        )
        expected = 0.2 * 0.75 + 0.8 * 0.75
        self.assertAlmostEqual(result[0]["combined_score"], expected)

    def test_keeps_only_best_chunk_per_source(self):
        candidates = [
            {"text": "weak a", "source": "A.pdf", "coarse_score": 0.9},
            {"text": "strong a", "source": "a.PDF", "coarse_score": 0.8},
            {"text": "b", "source": "b.pdf", "coarse_score": 0.7},
            {"text": "c", "source": "c.pdf", "coarse_score": 0.6},
            {"text": "d", "source": "d.pdf", "coarse_score": 0.5},
        ]
        result = fuse_scores_and_deduplicate_sources(
            candidates,
            [0.1, 0.9, 0.8, 0.7, 0.6],
            top_k=3,
        )
        self.assertEqual([item["source"] for item in result], ["a.PDF", "b.pdf", "c.pdf"])
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
