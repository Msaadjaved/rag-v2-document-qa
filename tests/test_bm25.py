import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import bm25, core


def recs(*texts):
    return [{"id": f"d:{i}", "document": "d", "index": i, "text": t} for i, t in enumerate(texts)]


MANUAL = recs(
    "E12 - Water tank empty. Fill the tank and restart.",
    "E47 - Grinder blocked by a foreign object. Unplug the machine and use the hexagonal key.",
    "E74 - Brewing unit missing. Insert the brewing unit.",
    "Descaling: use the recommended descaling liquid, never vinegar.",
    "Milk foam is best with milk at around 5 degrees Celsius.",
)


class TokenizerTests(unittest.TestCase):
    def test_compound_identifier_keeps_whole_and_parts(self):
        self.assertEqual(bm25.tokenize("KST-5031"), ["kst-5031", "kst", "5031"])

    def test_accents_and_case_are_folded(self):
        self.assertEqual(bm25.tokenize("Inès"), bm25.tokenize("ines"))

    def test_stopwords_removed_by_default(self):
        self.assertEqual(bm25.tokenize("What does E47 mean?"), ["e47", "mean"])

    def test_stopwords_can_be_disabled(self):
        bm25.set_stopwords(False)
        try:
            self.assertEqual(bm25.tokenize("What does E47 mean?"), ["what", "does", "e47", "mean"])
        finally:
            bm25.set_stopwords(True)

    def test_variants_use_different_index_files_and_fingerprints(self):
        a = bm25.index_path("own", 800, 10)
        bm25.set_stopwords(False)
        try:
            self.assertNotEqual(a, bm25.index_path("own", 800, 10))
        finally:
            bm25.set_stopwords(True)


class RankingTests(unittest.TestCase):
    def setUp(self):
        bm25.set_stopwords(True)
        self.index = bm25.BM25Index.build(MANUAL)

    def test_rare_identifier_wins(self):
        top = self.index.search("What does error E47 mean and how do I fix it?", 3)
        self.assertEqual(top[0][0], 1)

    def test_exact_code_not_confused_with_neighbour(self):
        e47 = self.index.search("E47", 5)
        e74 = self.index.search("E74", 5)
        self.assertEqual(e47[0][0], 1)
        self.assertEqual(e74[0][0], 2)

    def test_idf_rarer_is_heavier(self):
        idx = bm25.BM25Index.build(recs(
            "machine E47 grinder", "machine milk foam", "machine descaling liquid", "machine water tank"))
        self.assertGreater(idx.idf("e47"), idx.idf("machine"))

    def test_vocabulary_mismatch_returns_nothing_useful(self):
        # 'calcium' / 'deposits' appear nowhere: BM25 has no signal at all.
        self.assertEqual(self.index.search("calcium deposits", 5), [])

    def test_hyphenated_code_pins_exact_code(self):
        idx = bm25.BM25Index.build(recs(
            "KST-5031 Scheduled report not delivered, raised by postroom.",
            "KST-5038 Embed token expired.",
        ))
        self.assertEqual(idx.search("error KST-5031", 2)[0][0], 0)
        self.assertEqual(idx.search("error KST-5038", 2)[0][0], 1)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [
            mock.patch.object(core, "INDEX_DIR", Path(self.tmp.name)),
            mock.patch.dict(bm25._CACHE, clear=True),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_build_then_memory_then_disk(self):
        first = bm25.get_bm25("own", 800, 10, records=MANUAL)
        self.assertEqual(first.info["source"], "built")
        self.assertTrue(bm25.index_path("own", 800, 10).exists())
        self.assertEqual(bm25.get_bm25("own", 800, 10, records=MANUAL).info["source"], "memory")
        bm25._CACHE.clear()  # new process
        third = bm25.get_bm25("own", 800, 10, records=MANUAL)
        self.assertEqual(third.info["source"], "disk")
        self.assertEqual(third.search("E47", 1)[0]["id"], "d:1")

    def test_document_update_triggers_rebuild(self):
        bm25.get_bm25("own", 800, 10, records=MANUAL)
        updated = MANUAL + recs("E99 - New code added by an upload.")
        updated[-1]["id"] = "d:5"
        again = bm25.get_bm25("own", 800, 10, records=updated)
        self.assertEqual(again.info["source"], "built")
        self.assertEqual(again.search("E99", 1)[0]["id"], "d:5")

    def test_removed_document_disappears(self):
        bm25.get_bm25("own", 800, 10, records=MANUAL)
        fewer = MANUAL[:1]
        again = bm25.get_bm25("own", 800, 10, records=fewer)
        self.assertEqual(again.info["source"], "built")
        self.assertEqual(again.search("E47", 3), [])

    def test_corrupt_file_is_rebuilt(self):
        bm25.get_bm25("own", 800, 10, records=MANUAL)
        bm25._CACHE.clear()
        bm25.index_path("own", 800, 10).write_text("{not json", encoding="utf-8")
        self.assertEqual(bm25.get_bm25("own", 800, 10, records=MANUAL).info["source"], "built")

    def test_force_rebuild(self):
        bm25.get_bm25("own", 800, 10, records=MANUAL)
        self.assertEqual(bm25.get_bm25("own", 800, 10, records=MANUAL, force=True).info["source"], "built")

    def test_other_chunking_gets_its_own_file(self):
        bm25.get_bm25("own", 800, 10, records=MANUAL)
        bm25.get_bm25("own", 500, 20, records=MANUAL)
        self.assertNotEqual(bm25.index_path("own", 800, 10), bm25.index_path("own", 500, 20))

    def test_scores_zero_chunks_are_not_returned(self):
        r = bm25.get_bm25("own", 800, 10, records=MANUAL)
        self.assertTrue(all(h["score"] > 0 for h in r.search("grinder tank", 10)))


if __name__ == "__main__":
    unittest.main()
