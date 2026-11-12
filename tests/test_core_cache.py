"""Dense retriever caching, with a fake embedding model (no download needed)."""
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

import numpy as np

from app import core


class FakeModel:
    """Hashed bag-of-words embeddings: deterministic, offline."""
    calls = 0

    def encode(self, texts, normalize_embeddings=True):
        FakeModel.calls += 1
        out = np.zeros((len(texts), 64))
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, zlib.crc32(w.encode()) % 64] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


def recs(*texts):
    return [{"id": f"d:{i}", "document": "d", "index": i, "text": t} for i, t in enumerate(texts)]


class DenseCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [
            mock.patch.object(core, "INDEX_DIR", Path(self.tmp.name)),
            mock.patch.object(core, "get_embedding_model", lambda: FakeModel()),
            mock.patch.dict(core._DENSE_CACHE, clear=True),
        ]
        for p in self.patches:
            p.start()
        FakeModel.calls = 0

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_fingerprint_changes_with_text_and_order(self):
        a = core.records_fingerprint(recs("x", "y"))
        self.assertEqual(a, core.records_fingerprint(recs("x", "y")))
        self.assertNotEqual(a, core.records_fingerprint(recs("y", "x")))
        self.assertNotEqual(a, core.records_fingerprint(recs("x", "z")))

    def test_in_memory_cache_reuses_retriever(self):
        r = recs("alpha beta", "gamma delta")
        r1 = core.get_dense("c", 800, 10, records=r)
        calls = FakeModel.calls
        r2 = core.get_dense("c", 800, 10, records=r)
        self.assertIs(r1, r2)
        self.assertEqual(FakeModel.calls, calls)

    def test_disk_cache_survives_process_restart_and_invalidates(self):
        r = recs("alpha beta", "gamma delta")
        core.get_dense("c", 800, 10, records=r)
        core._DENSE_CACHE.clear()  # simulate a new process
        FakeModel.calls = 0
        again = core.get_dense("c", 800, 10, records=r)
        self.assertTrue(again.cache_hit)
        self.assertEqual(FakeModel.calls, 0)  # nothing re-encoded
        changed = core.get_dense("c", 800, 10, records=recs("alpha beta", "NEW text"))
        self.assertFalse(changed.cache_hit)

    def test_search_ranks_matching_chunk_first(self):
        r = core.get_dense("c", 800, 10, records=recs("grinder blocked foreign object", "milk foam temperature"))
        self.assertEqual(r.search("milk foam", 2)[0]["id"], "d:1")

    def test_strict_refuses_tfidf_fallback(self):
        def boom():
            raise ImportError("no model")
        with mock.patch.object(core, "get_embedding_model", boom):
            with self.assertRaises(RuntimeError):
                core.Retriever(recs("grinder blocked"), strict=True)
            lenient = core.Retriever(recs("grinder blocked", "milk foam"))
            self.assertEqual(lenient.backend, "tfidf")


if __name__ == "__main__":
    unittest.main()
