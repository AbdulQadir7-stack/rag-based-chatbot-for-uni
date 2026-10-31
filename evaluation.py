# evaluation.py - Retrieval and answer quality evaluation

import argparse
import json
from pathlib import Path
from statistics import mean
from collections import defaultdict

from agents import get_collection_for_query, generate_answer, retrieve_documents_federated


def load_dataset(path: str) -> list[dict]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if dataset_path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    return json.loads(dataset_path.read_text(encoding="utf-8"))


def doc_matches(doc: dict, expected: dict) -> bool:
    expected_id = expected.get("expected_doc_id")
    expected_text = (expected.get("expected_text") or "").lower()
    expected_source = expected.get("expected_source")
    expected_collection = expected.get("collection")

    if expected_id and doc.get("id") == expected_id:
        return True
    if expected_source and doc.get("metadata", {}).get("source") == expected_source:
        return True
    if expected_collection and doc.get("metadata", {}).get("collection") == expected_collection:
        # Use "query" field (standard in eval_dataset.json); fall back to "expected_question"
        question = expected.get("query") or expected.get("expected_question") or ""
        if question.lower() in doc.get("text", "").lower():
            return True
    if expected_text and expected_text in doc.get("text", "").lower():
        return True

    return False


def reciprocal_rank(results: list[dict], expected: dict) -> float:
    for index, doc in enumerate(results, start=1):
        if doc_matches(doc, expected):
            return 1 / index
    return 0.0


def evaluate_retrieval(rows: list[dict], top_k: int) -> tuple[list[dict], dict]:
    detailed = []

    for row in rows:
        query = row["query"]
        collection = row.get("collection") or get_collection_for_query(query)
        docs = retrieve_documents_federated(query, n_results=top_k)
        rr = reciprocal_rank(docs, row)

        detailed.append({
            "query": query,
            "collection": collection or "none",
            "hit_at_1": 1.0 if docs and doc_matches(docs[0], row) else 0.0,
            "reciprocal_rank": rr,
            "top_doc_id": docs[0]["id"] if docs else "",
            "top_rerank_score": docs[0].get("rerank_score", 0.0) if docs else 0.0,
            "rerank_method": docs[0].get("rerank_method", "") if docs else "",
        })

    metrics = {
        "HitRate@1": round(mean(item["hit_at_1"] for item in detailed), 4) if detailed else 0.0,
        "MRR": round(mean(item["reciprocal_rank"] for item in detailed), 4) if detailed else 0.0,
        "count": len(detailed),
    }
    by_collection = defaultdict(list)
    for item in detailed:
        by_collection[item["collection"]].append(item)

    metrics["by_collection"] = {
        collection: {
            "HitRate@1": round(mean(item["hit_at_1"] for item in items), 4),
            "MRR": round(mean(item["reciprocal_rank"] for item in items), 4),
            "count": len(items),
        }
        for collection, items in sorted(by_collection.items())
    }
    return detailed, metrics


def evaluate_bertscore(rows: list[dict], top_k: int) -> dict:
    try:
        from bert_score import score
    except ImportError as exc:
        raise RuntimeError("Install bert-score to compute BERTScore: pip install bert-score") from exc

    predictions = []
    references = []

    for row in rows:
        reference = row.get("reference_answer")
        if not reference:
            continue

        collection = row.get("collection") or get_collection_for_query(row["query"])
        docs = retrieve_documents_federated(row["query"], n_results=top_k)
        answer = generate_answer(row["query"], [doc["text"] for doc in docs])
        predictions.append(answer)
        references.append(reference)

    if not predictions:
        return {"BERTScore_P": 0.0, "BERTScore_R": 0.0, "BERTScore_F1": 0.0, "count": 0}

    precision, recall, f1 = score(predictions, references, lang="en", verbose=False)
    return {
        "BERTScore_P": round(float(precision.mean()), 4),
        "BERTScore_R": round(float(recall.mean()), 4),
        "BERTScore_F1": round(float(f1.mean()), 4),
        "count": len(predictions),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate AJKU RAG retrieval quality.")
    parser.add_argument("--dataset", default="eval_dataset.json", help="JSON or JSONL evaluation dataset")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved docs to score")
    parser.add_argument("--bertscore", action="store_true", help="Also generate answers and compute BERTScore")
    parser.add_argument("--output", default="", help="Optional path for detailed JSON results")
    args = parser.parse_args()

    rows = load_dataset(args.dataset)
    detailed, metrics = evaluate_retrieval(rows, args.top_k)

    if args.bertscore:
        metrics.update(evaluate_bertscore(rows, args.top_k))

    print(json.dumps(metrics, indent=2))
    print("\n| Metric | Value |")
    print("|---|---:|")
    for key, value in metrics.items():
        if key != "by_collection":
            print(f"| {key} | {value} |")

    if metrics.get("by_collection"):
        print("\n| Collection | HitRate@1 | MRR | Count |")
        print("|---|---:|---:|---:|")
        for collection, values in metrics["by_collection"].items():
            print(
                f"| {collection} | {values['HitRate@1']} | "
                f"{values['MRR']} | {values['count']} |"
            )

    if args.output:
        Path(args.output).write_text(
            json.dumps({"metrics": metrics, "details": detailed}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
