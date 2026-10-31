# debug_collections.py - See what's stored in each ChromaDB collection

import chromadb

client = chromadb.PersistentClient(path="./chroma_db_lite")

COLLECTIONS = [
    "ajku_admission",
    "ajku_scholarship",
    "ajku_exam",
    "ajku_academic",
    "ajku_complaint",
]

print("=" * 60)
print("ChromaDB Collection Inspector")
print("=" * 60)

for coll_name in COLLECTIONS:
    try:
        coll = client.get_collection(coll_name)
        docs = coll.get()["documents"]
        print(f"\n📂 {coll_name} — {len(docs)} chunks")
        print("-" * 40)
        for i, doc in enumerate(docs[:3]):  # show first 3
            print(f"  [{i+1}] {doc[:120]}...")
        if len(docs) > 3:
            print(f"  ... and {len(docs)-3} more")
    except Exception as e:
        print(f"\n❌ {coll_name}: NOT FOUND — Run populate_multicol.py first")

print("\n" + "=" * 60)
