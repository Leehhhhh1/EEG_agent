import unittest

from RAG.ranking import (
    fuse_scores_and_select_top_k,
    fuse_three_way_ranks,
    reciprocal_rank_fusion,
)


class RAGRankingTests(unittest.TestCase):
    def test_three_way_fusion_rewards_candidate_supported_by_all_modes(self):
        candidates = [
            {"chunk_id": "a", "dense_rank": 1, "sparse_rank": 1},
            {"chunk_id": "b", "dense_rank": 2},
            {"chunk_id": "c", "sparse_rank": 2},
        ]
        result = fuse_three_way_ranks(candidates, [0.8, 0.9, 0.7], top_k=3)
        self.assertEqual(result[0]["chunk_id"], "a")
        self.assertIn("colbert_rank", result[0])
        self.assertIn("three_way_score", result[0])

    def test_reciprocal_rank_fusion_combines_dense_and_sparse_candidates(self):
        dense = [
            {"chunk_id": "a", "text": "a", "dense_score": 0.8},
            {"chunk_id": "b", "text": "b", "dense_score": 0.7},
        ]
        sparse = [
            {"chunk_id": "b", "text": "b", "sparse_score": 2.0},
            {"chunk_id": "c", "text": "c", "sparse_score": 1.0},
        ]
        result = reciprocal_rank_fusion(dense, sparse)
        self.assertEqual(result[0]["chunk_id"], "b")
        self.assertIn("dense_rank", result[0])
        self.assertIn("sparse_rank", result[0])

    def test_uses_point_two_point_eight_weighting(self):
        result = fuse_scores_and_select_top_k(
            [{"text": "a", "source": "a.pdf", "coarse_score": 0.5}],
            [0.75],
        )
        expected = 0.2 * 0.75 + 0.8 * 0.75
        self.assertAlmostEqual(result[0]["combined_score"], expected)

    def test_allows_multiple_top_chunks_from_the_same_source(self):
        candidates = [
            {"text": "weak a", "source": "A.pdf", "coarse_score": 0.9},
            {"text": "strong a", "source": "a.PDF", "coarse_score": 0.8},
            {"text": "b", "source": "b.pdf", "coarse_score": 0.7},
            {"text": "c", "source": "c.pdf", "coarse_score": 0.6},
            {"text": "d", "source": "d.pdf", "coarse_score": 0.5},
        ]
        result = fuse_scores_and_select_top_k(
            candidates,
            [0.75, 0.9, 0.8, 0.7, 0.6],
            top_k=3,
        )
        self.assertEqual([item["source"] for item in result], ["a.PDF", "b.pdf", "A.pdf"])
        self.assertEqual([item["text"] for item in result], ["strong a", "b", "weak a"])
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
