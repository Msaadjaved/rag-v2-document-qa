import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import bm25, core, retrieval
from app.fusion import rrf


def hit(i, score=1.0):
    return {"id": f"c{i}", "document": "d", "index": i, "text": f"t{i}", "score": score}


class RrfTests(unittest.TestCase):
    def test_formula(self):
        fused = rrf({"a": [hit(1), hit(2)], "b": [hit(2), hit(3)]}, k=60)
        scores = {h["id"]: h["score"] for h in fused}
        self.assertAlmostEqual(scores["c2"], 1 / 62 + 1 / 61)
        self.assertAlmostEqual(scores["c1"], 1 / 61)
        self.assertAlmostEqual(scores["c3"], 1 / 62)

    def test_chunk_found_by_both_beats_chunks_found_by_one(self):
        fused = rrf({"a": [hit(1), hit(2)], "b": [hit(3), hit(2)]})
        self.assertEqual(fused[0]["id"], "c2")

    def test_sources_report_retriever_and_rank(self):
        fused = rrf({"dense": [hit(1, 0.9), hit(2, 0.5)], "bm25": [hit(2, 7.0)]})
        c2 = next(h for h in fused if h["id"] == "c2")
        self.assertEqual(c2["sources"], {"dense": {"rank": 2, "score": 0.5}, "bm25": {"rank": 1, "score": 7.0}})

    def test_weights_change_order(self):
        lists = {"a": [hit(1)], "b": [hit(2)]}
        self.assertEqual(rrf(lists)[0]["id"], "c1")  # tie -> id order
        self.assertEqual(rrf(lists, weights={"b": 2.0})[0]["id"], "c2")

    def test_deterministic_ties_and_limit(self):
        fused = rrf({"a": [hit(1)], "b": [hit(2)]}, limit=1)
        self.assertEqual([h["id"] for h in fused], ["c1"])

    def test_empty(self):
        self.assertEqual(rrf({"a": [], "b": []}), [])


RECORDS = [
    {"id": "d:0", "document": "d", "index": 0, "text": "E12 water tank empty"},
    {"id": "d:1", "document": "d", "index": 1, "text": "E47 grinder blocked foreign object"},
    {"id": "d:2", "document": "d", "index": 2, "text": "limescale descaling programme"},
]


class FakeDense:
    backend = "fake-dense"

    def search(self, query, k):
        order = [2, 0, 1]  # dense "prefers" the descaling chunk
        return [{**RECORDS[i], "score": 1.0 - 0.1 * r} for r, i in enumerate(order)][:k]


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [
            mock.patch.object(core, "INDEX_DIR", Path(self.tmp.name)),
            mock.patch.dict(bm25._CACHE, clear=True),
            mock.patch.object(core, "get_dense", lambda *a, **k: FakeDense()),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def run_retrieve(self, retrievers, **kw):
        return retrieval.retrieve("own", "error E47", 800, 10, retrievers, records=RECORDS, **kw)

    def test_single_retriever_is_not_fused(self):
        out = self.run_retrieve(["dense"], depth=3)
        self.assertEqual([h["id"] for h in out["hits"]], ["d:2", "d:0", "d:1"])
        self.assertEqual(out["hits"][0]["sources"], {"dense": {"rank": 1, "score": 1.0}})

    def test_bm25_alone_finds_identifier(self):
        out = self.run_retrieve(["bm25"], depth=3)
        self.assertEqual(out["hits"][0]["id"], "d:1")
        self.assertEqual([h["id"] for h in out["hits"]], ["d:1"])  # zero-score chunks excluded

    def test_hybrid_rescues_chunk_dense_ranked_last(self):
        out = self.run_retrieve(["dense", "bm25"], depth=3)
        top = out["hits"][0]
        self.assertEqual(top["id"], "d:1")
        self.assertEqual(set(top["sources"]), {"dense", "bm25"})
        self.assertEqual(top["sources"]["dense"]["rank"], 3)
        self.assertEqual(top["sources"]["bm25"]["rank"], 1)

    def test_graph_rejected_on_own_corpus(self):
        with self.assertRaises(ValueError):
            retrieval.retrieve("own", "q", 800, 10, ["graph"], records=RECORDS)

    def test_no_retriever_rejected(self):
        with self.assertRaises(ValueError):
            self.run_retrieve([])

    def test_unknown_retriever_rejected(self):
        with self.assertRaises(ValueError):
            self.run_retrieve(["colbert"])

    def test_graph_missing_module_is_reported(self):
        with mock.patch.object(retrieval, "_run_graph", side_effect=retrieval.RetrieverUnavailable("x")):
            with self.assertRaises(retrieval.RetrieverUnavailable):
                retrieval.retrieve("kestrel", "q", 800, 10, ["graph"], records=RECORDS)

    def test_graph_hits_path_and_usage_flow_through(self):
        fake = {
            "hits": [{"id": "d:2", "score": 3.0}, {"id": "unknown:9", "score": 2.0}],
            "path": {"entities": ["x"], "edges": [], "chunks": ["d:2"]},
            "usage": {"llm_calls": 1, "prompt_tokens": 50, "completion_tokens": 5},
        }
        with mock.patch.object(retrieval, "_run_graph", return_value=fake):
            out = retrieval.retrieve("kestrel", "q", 800, 10, ["dense", "bm25", "graph"],
                                     records=RECORDS, depth=3)
        self.assertEqual(out["graph_path"]["entities"], ["x"])
        self.assertEqual(out["usage"]["llm_calls"], 1)
        self.assertTrue(all(h["id"] != "unknown:9" for h in out["hits"]))
        d2 = next(h for h in out["hits"] if h["id"] == "d:2")
        self.assertIn("graph", d2["sources"])

    def test_default_retrievers_env(self):
        with mock.patch.dict("os.environ", {"DEFAULT_RETRIEVERS": "bm25, graph, nonsense"}):
            self.assertEqual(retrieval.default_retrievers(), ["bm25", "graph"])
        with mock.patch.dict("os.environ", {"DEFAULT_RETRIEVERS": "x"}):
            self.assertEqual(retrieval.default_retrievers(), ["dense", "bm25"])


if __name__ == "__main__":
    unittest.main()
