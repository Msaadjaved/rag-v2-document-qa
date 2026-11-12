import unittest
from unittest import mock

from app import evaluation as ev

RECORDS = [
    {"id": "runbook:0", "document": "runbook.md", "index": 0, "text": "KST-5031 raised by postroom. escalate to the team that owns postroom"},
    {"id": "catalog:0", "document": "catalog.md", "index": 0, "text": "| Messaging | Ines | #oncall-messaging | 08:00 |"},
    {"id": "guide:0", "document": "guide.md", "index": 0, "text": "Annual billing receives a 20% discount."},
    {"id": "guide:1", "document": "guide.md", "index": 1, "text": "unrelated filler text"},
]

QUESTIONS = [
    {"id": "q08", "question": "KST-5031 who?", "type": "two_hop", "answerable": True,
     "evidence": [{"source": "runbook.md", "quote": "escalate to the team that owns postroom"},
                  {"source": "catalog.md", "quote": "| Messaging | Ines | #oncall-messaging"}]},
    {"id": "q07", "question": "annual discount?", "type": "versions", "answerable": True,
     "evidence": [{"source": "guide.md", "quote": "20% discount"}]},
    {"id": "q09", "question": "iphone app?", "type": "not_in_corpus", "answerable": False, "evidence": []},
    {"id": "OWN-1", "question": "vocab?", "type": "different_vocabulary", "answerable": True, "answer_quote": "unrelated filler"},
    {"id": "OWN-2", "question": "bad quote?", "type": "two_hops", "answerable": True, "answer_quote": "this quote is nowhere"},
]

RANKINGS = {
    "KST-5031 who?": ["runbook:0", "guide:1", "guide:0", "catalog:0"],   # hop1 @1, hop2 @4
    "annual discount?": ["guide:1", "guide:0"],                           # gold @2
    "iphone app?": ["guide:1"],
    "vocab?": ["guide:1"],
    "bad quote?": ["guide:0"],
}


def fake_retrieve(corpus, question, size, overlap, retrievers, depth=5, **kw):
    ids = RANKINGS[question][:depth]
    by = {r["id"]: r for r in RECORDS}
    return {
        "hits": [{**by[i], "score": 1.0 / (n + 1),
                  "sources": {retrievers[0]: {"rank": n + 1, "score": 1.0 / (n + 1)}}}
                 for n, i in enumerate(ids)],
        "retrievers": retrievers, "graph_path": None, "backend": {},
        "timings": {"dense_s": 0.01, "fusion_s": 0.0, "total_s": 0.02},
        "usage": {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
    }


def run(top_k=5, retrievers=("dense",), questions=QUESTIONS):
    with mock.patch.object(ev, "retrieve", fake_retrieve), mock.patch.object(ev, "_warm", lambda *a, **k: {}):
        return ev.evaluate_corpus("kestrel", 800, 10, top_k, list(retrievers),
                                  questions=questions, records=RECORDS)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.res = run()
        self.rows = {r["id"]: r for r in self.res["results"]}
        self.s = self.res["summary"]

    def test_two_hop_recall_counts_every_hop(self):
        q = self.rows["q08"]
        self.assertTrue(q["hit_at_1"])
        self.assertEqual(q["recall_at_1"], 0.5)   # only hop 1 in top 1
        self.assertEqual(q["recall_at_3"], 0.5)
        self.assertEqual(q["recall_at_5"], 1.0)   # hop 2 appears at rank 4
        self.assertEqual(q["recall_at_k"], 1.0)

    def test_hit_is_at_least_one_hop_and_nested(self):
        for r in self.res["results"]:
            if r["answerable"]:
                flags = [r[f"hit_at_{n}"] for n in ev.CUTOFFS]
                self.assertEqual(flags, sorted(flags))  # False... then True: nested

    def test_not_in_corpus_excluded_from_metrics(self):
        q = self.rows["q09"]
        self.assertIsNone(q["hit_at_5"])
        self.assertIsNone(q["recall_at_5"])
        self.assertEqual(self.s["answerable_questions"], 4)
        self.assertEqual(self.s["total_questions"], 5)
        self.assertEqual(self.s["by_type"]["not_in_corpus"]["count"], 0)
        self.assertEqual(self.s["by_type"]["not_in_corpus"]["total"], 1)

    def test_type_aliases_kept_raw_value(self):
        self.assertEqual(self.rows["OWN-1"]["type"], "vocabulary")
        self.assertEqual(self.rows["OWN-1"]["type_raw"], "different_vocabulary")
        self.assertEqual(self.rows["OWN-2"]["type"], "two_hop")
        self.assertNotIn("different_vocabulary", self.s["by_type"])

    def test_legacy_answer_quote_is_one_hop(self):
        self.assertEqual(len(self.rows["OWN-1"]["hops"]), 1)
        self.assertEqual(self.rows["OWN-1"]["hops"][0]["gold_chunk_ids"], ["guide:1"])

    def test_unmatched_quote_is_flagged_and_counts_as_miss(self):
        q = self.rows["OWN-2"]
        self.assertEqual(q["unmatched_hops"], ["this quote is nowhere"])
        self.assertFalse(q["hit_at_10"])
        self.assertEqual(q["recall_at_10"], 0.0)  # unmatched hop = not retrieved (and flagged)
        self.assertIn("OWN-2", self.s["questions_with_unmatched_hops"])

    def test_first_gold_rank_and_found_by(self):
        self.assertEqual(self.rows["q07"]["first_gold_rank"], 2)
        hop2 = self.rows["q08"]["hops"][1]
        self.assertEqual(hop2["retrieved_by"], {"dense": 4})

    def test_by_type_and_overall_means(self):
        two = self.s["by_type"]["two_hop"]
        self.assertEqual(two["count"], 2)                     # q08 + OWN-2
        self.assertAlmostEqual(two["hit_rate_at_5"], 0.5)     # q08 hit, OWN-2 miss
        self.assertAlmostEqual(two["recall_at_1"], 0.25)      # q08 0.5, OWN-2 0.0 (unmatched hop)
        self.assertAlmostEqual(self.s["hit_rates"]["hit_rate_at_1"], 2 / 4)  # q08, OWN-1(guide:1 at 1)

    def test_cost_per_question_present(self):
        c = self.s["cost_per_question"]
        self.assertAlmostEqual(c["retrieval_ms"], 20.0)
        self.assertEqual(c["llm_calls"], 0)

    def test_top_k_not_in_standard_cutoffs(self):
        res = run(top_k=4)
        row = {r["id"]: r for r in res["results"]}["q08"]
        self.assertEqual(row["recall_at_k"], 1.0)  # hop 2 at rank 4
        self.assertEqual(len(row["retrieved"]), 4)

    def test_regressions_finds_worse_ranking(self):
        worse_rank = {**RANKINGS, "annual discount?": ["guide:1", "runbook:0", "guide:0"]}
        with mock.patch.dict(RANKINGS, worse_rank):
            new = run()
        out = ev.regressions(self.res, new)
        self.assertEqual([r["id"] for r in out], ["q07"])
        self.assertEqual((out[0]["rank_before"], out[0]["rank_after"]), (2, 3))

    def test_regressions_none_when_equal(self):
        self.assertEqual(ev.regressions(self.res, run()), [])


if __name__ == "__main__":
    unittest.main()
