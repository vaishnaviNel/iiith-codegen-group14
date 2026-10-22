"""Build both retrieval indexes and run a sanity query.

Run from the repo root:  python -m scripts.build_index
"""

from src.rag.corpus import codoc_corpus, mbpp_corpus
from src.rag.retriever import Retriever

for name, make in [("mbpp", mbpp_corpus), ("codoc", codoc_corpus)]:
    entries = make()
    print(f"[{name}] {len(entries)} corpus entries")
    Retriever.build(entries).save(f"indexes/{name}")

r = Retriever.load("indexes/mbpp")
for hit in r.search("write a function to find the shortest string in a list", k=3):
    print(f"{hit['score']:.3f}  {hit['id']}  {hit['query']}")
