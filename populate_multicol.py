# populate_multicol.py - Load FAQ data into ChromaDB collections

import chromadb
from chromadb.utils import embedding_functions
import os
import re
import sys

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db_lite")

# Map file names to collection names
FILES_TO_COLLECTIONS = {
    "admission fee faqs.txt": "ajku_admission",
    "acdemic faqs.txt":       "ajku_academic",
    "scholarships faqs.txt":  "ajku_scholarship",
    "exam faqs.txt":          "ajku_exam",
    "complaint faqs.txt":     "ajku_complaint",
}

DATA_DIR = "data"

def populate():
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: '{DATA_DIR}' folder not found!")
        print("   Create a 'data' folder and put your .txt FAQ files inside it.")
        sys.exit(1)

    total_added = 0

    for filename, coll_name in FILES_TO_COLLECTIONS.items():
        filepath = os.path.join(DATA_DIR, filename)

        if not os.path.exists(filepath):
            print(f"WARNING: {filepath} not found - skipping {coll_name}")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Split on either "Question:" blocks or numbered FAQ items like "1. ..."
        blocks = re.split(r'\n(?=(?:Question:|\d+\.\s))', content)
        docs = [
            b.strip()
            for b in blocks
            if b.strip() and "Answer:" in b and len(b.strip()) > 10
        ]

        if not docs:
            print(f"WARNING: No valid content in {filename} - skipping")
            continue

        # Delete old collection if it exists
        try:
            client.delete_collection(coll_name)
        except Exception:
            pass

        # Create fresh collection
        collection = client.create_collection(
            name=coll_name,
            embedding_function=embedding_fn
        )

        ids = [f"{coll_name}_{i}" for i in range(len(docs))]
        metadatas = []
        for i, doc in enumerate(docs):
            question_match = re.search(r"Question:\s*(.+)", doc, flags=re.IGNORECASE)
            if not question_match:
                question_match = re.search(r"^\d+\.\s*(.+)", doc)
            metadatas.append({
                "source": filename,
                "collection": coll_name,
                "question": question_match.group(1).strip() if question_match else f"{coll_name} FAQ {i + 1}",
            })
        collection.add(documents=docs, ids=ids, metadatas=metadatas)

        print(f"OK: {coll_name}: {len(docs)} chunks added")
        total_added += len(docs)

    print(f"\nDone. Total {total_added} chunks added to ChromaDB.")


if __name__ == "__main__":
    populate()
