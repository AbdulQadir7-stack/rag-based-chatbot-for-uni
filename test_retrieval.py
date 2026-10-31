# test_retrieval.py - Test ChromaDB vector search

import chromadb
from chromadb.utils import embedding_functions

PERSIST_DIR = "./chroma_db_lite"
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=PERSIST_DIR)

test_cases = [
    ("ajku_admission",   "admission last date kab hai"),
    ("ajku_scholarship", "PEEF scholarship ke liye apply kaise karein"),
    ("ajku_exam",        "exam date sheet kab aayegi"),
    ("ajku_academic",    "BS Computer Science program hai"),
    ("ajku_complaint",   "complaint kahan submit karein"),
]

print("=" * 60)
print("Retrieval Test Results")
print("=" * 60)

for coll_name, query in test_cases:
    print(f"\n🔍 Query: '{query}'")
    print(f"   Collection: {coll_name}")
    try:
        coll = client.get_collection(coll_name, embedding_function=embedding_fn)
        results = coll.query(query_texts=[query], n_results=2)
        docs = results["documents"][0]
        dists = results["distances"][0]
        for i, (doc, dist) in enumerate(zip(docs, dists)):
            print(f"   Result {i+1} (dist={dist:.3f}): {doc[:120]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
