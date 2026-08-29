import unittest

from RAG.ranking import filter_by_rerank_threshold, passes_faiss_probe
from RAG.retrieval_policy import build_retrieval_query, decide_retrieval


class RetrievalPolicyTests(unittest.TestCase):
    def test_clear_intents_use_deterministic_routes(self):
        self.assertEqual(
            decide_retrieval("什么是 LPD？", has_eeg_session=False).mode,
            "retrieve",
        )
        self.assertEqual(
            decide_retrieval("你好", has_eeg_session=False).mode,
            "skip",
        )
        decision = decide_retrieval(
            "前60秒有没有癫痫样放电？",
            has_eeg_session=True,
        )
        self.assertEqual(decision.mode, "skip")
        self.assertEqual(decision.reason, "recording_specific_tool_query")

    def test_ambiguous_intent_uses_semantic_probe(self):
        decision = decide_retrieval("睡眠阶段中的变化", has_eeg_session=False)
        self.assertEqual(decision.mode, "probe")

    def test_follow_up_prepends_only_the_previous_user_question(self):
        query = build_retrieval_query(
            "它和 GPD 有什么区别？",
            "什么是 LPD？",
        )
        self.assertEqual(query, "什么是 LPD？\n它和 GPD 有什么区别？")
        self.assertEqual(
            build_retrieval_query("什么是 GPD？", "什么是 LPD？"),
            "什么是 GPD？",
        )

    def test_faiss_probe_blocks_low_relevance_before_reranking(self):
        candidates = [{"coarse_score": 0.34}]
        self.assertFalse(passes_faiss_probe(candidates, threshold=0.35))
        candidates[0]["coarse_score"] = 0.35
        self.assertTrue(passes_faiss_probe(candidates, threshold=0.35))

    def test_rerank_threshold_returns_dynamic_zero_to_three(self):
        ranked = [
            {"text": "a", "rerank_score": 0.9},
            {"text": "b", "rerank_score": 0.6},
            {"text": "c", "rerank_score": 0.49},
            {"text": "d", "rerank_score": 0.8},
        ]
        selected = filter_by_rerank_threshold(ranked, threshold=0.5, top_k=3)
        self.assertEqual([item["text"] for item in selected], ["a", "b", "d"])
        self.assertEqual(
            filter_by_rerank_threshold(ranked, threshold=0.95, top_k=3),
            [],
        )


if __name__ == "__main__":
    unittest.main()
