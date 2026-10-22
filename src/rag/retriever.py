"""Dense retriever: embed every corpus query once, then top-k by cosine similarity.

This is the whole "R" in RAG. With <10k entries a numpy matrix-multiply is exact and
instant on CPU, so no vector DB / FAISS is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 33M params, runs fine on a Mac CPU
# bge was trained so that *search queries* carry this prefix; documents do not.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Retriever:
    def __init__(self, entries: List[Dict], vectors: np.ndarray, model: SentenceTransformer):
        self.entries = entries
        self.vectors = vectors  # (N, d), L2-normalised -> dot product == cosine
        self.model = model

    @classmethod
    def build(cls, entries: List[Dict], model_name: str = EMBED_MODEL) -> "Retriever":
        model = SentenceTransformer(model_name)
        vectors = model.encode(
            [e["query"] for e in entries],
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=True,
        )
        return cls(entries, np.asarray(vectors, dtype=np.float32), model)

    def search(self, text: str, k: int = 3) -> List[Dict]:
        q = self.model.encode([QUERY_PREFIX + text], normalize_embeddings=True)[0]
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [{**self.entries[i], "score": float(scores[i])} for i in top]

    # -- persistence: build once, reload instantly ---------------------------------
    def save(self, directory: str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        (d / "entries.json").write_text(json.dumps(self.entries), encoding="utf-8")

    @classmethod
    def load(cls, directory: str, model_name: str = EMBED_MODEL) -> "Retriever":
        d = Path(directory)
        entries = json.loads((d / "entries.json").read_text(encoding="utf-8"))
        return cls(entries, np.load(d / "vectors.npy"), SentenceTransformer(model_name))
