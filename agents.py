# agents.py - Advanced Multi-Agent Hybrid RAG Pipeline with Gemini AI

import chromadb
from chromadb.utils import embedding_functions
from google import genai
import os
import difflib
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from retriever import hybrid_rerank_search, hybrid_search, rerank

# ─────────────────────────────────────────────
# LOAD ENV VARIABLES
# ─────────────────────────────────────────────

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

GEMINI_MODEL_CANDIDATES = [
    os.getenv("GEMINI_MODEL", "").strip(),
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-flash-lite-latest",
]
GEMINI_MODEL_CANDIDATES.extend(
    model.strip()
    for model in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",")
    if model.strip()
)
GEMINI_MODEL_CANDIDATES = list(dict.fromkeys(
    model.replace("models/", "")
    for model in GEMINI_MODEL_CANDIDATES
    if model
))
gemini_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None
available_gemini_models_cache: set[str] | None = None
GEMINI_MAX_RETRIES = max(1, int(os.getenv("GEMINI_MAX_RETRIES", "2")))
GEMINI_BACKOFF_SECONDS = max(0.1, float(os.getenv("GEMINI_BACKOFF_SECONDS", "0.75")))
GEMINI_UNAVAILABLE_MESSAGE = (
    "Gemini service is temporarily busy. I found relevant AJKU information, "
    "but could not generate a polished answer right now. Please try again in a moment."
)


def get_gemini_client():
    if gemini_client is None:
        raise RuntimeError("GOOGLE_API_KEY is missing. Add it to your .env file.")
    return gemini_client


def get_available_gemini_models() -> set[str]:
    global available_gemini_models_cache

    if available_gemini_models_cache is not None:
        return available_gemini_models_cache

    try:
        models = set()
        for model in get_gemini_client().models.list():
            methods = (
                getattr(model, "supported_actions", None)
                or getattr(model, "supported_generation_methods", [])
                or []
            )
            if "generateContent" in methods:
                models.add(model.name.replace("models/", ""))
        available_gemini_models_cache = models
    except Exception as exc:
        print(f"Gemini model list unavailable: {summarize_gemini_error(exc)}")
        available_gemini_models_cache = set(GEMINI_MODEL_CANDIDATES)

    return available_gemini_models_cache


def get_valid_gemini_models() -> list[str]:
    available = get_available_gemini_models()
    valid = [model for model in GEMINI_MODEL_CANDIDATES if model in available]
    return valid or GEMINI_MODEL_CANDIDATES[:1]


def summarize_gemini_error(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    status = getattr(exc, "status", None)
    message = str(exc).splitlines()[0]
    parts = []
    if status_code:
        parts.append(str(status_code))
    if status:
        parts.append(str(status))
    parts.append(message[:180])
    return " ".join(parts)


def is_retryable_gemini_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    text = str(exc).upper()
    return (
        status_code in {429, 500, 502, 503, 504}
        or "RESOURCE_EXHAUSTED" in text
        or "UNAVAILABLE" in text
        or "TOO_MANY_REQUESTS" in text
        or "429" in text
        or "503" in text
    )


def context_fallback_answer(context_chunks: list[str]) -> str:
    if not context_chunks:
        return "Server error."
    return f"{GEMINI_UNAVAILABLE_MESSAGE}\n\nRelevant retrieved context:\n{context_chunks[0]}"


def sleep_before_retry(attempt: int):
    time.sleep(GEMINI_BACKOFF_SECONDS * (2 ** attempt))

# ─────────────────────────────────────────────
# EMBEDDING + CHROMADB SETUP
# ─────────────────────────────────────────────

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    "all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(
    path="./chroma_db_lite"
)

docs_cache: dict[str, list[dict]] = {}
vocab_cache: set[str] | None = None
question_cache: list[dict] | None = None
LOW_CONFIDENCE_MESSAGE = "Sorry, I couldn't find this in AJKU data."
MIN_CONFIDENCE_SCORE = 20

# ─────────────────────────────────────────────
# COLLECTION ROUTING KEYWORDS
# ─────────────────────────────────────────────

COLLECTION_KEYWORDS = {

    "ajku_admission": [
        "admission",
        "form",
        "eligibility",
        "entry test",
        "last date",
        "prospectus",
        "fee",
        "dues",
        "hostel fee",
        "concession",
        "installment",
        "uni fee",
        "merit list",
        "apply",
        "application",
        "lat",
        "mdcat",
        "ugat",
        "quota",
        "seats"
    ],

    "ajku_scholarship": [
        "scholarship",
        "financial aid",
        "peef",
        "muslim hands",
        "need-based",
        "hec",
        "hbl",
        "stipend",
        "grant",
        "income limit"
    ],

    "ajku_exam": [
        "exam",
        "result",
        "date sheet",
        "supply",
        "practical",
        "roll number",
        "mid term",
        "final term",
        "paper",
        "marks",
        "gpa",
        "cgpa",
        "assessment",
        "test"
    ],

    "ajku_academic": [
        "program",
        "course",
        "department",
        "faculty",
        "campus",
        "hostel",
        "library",
        "sport",
        "transport",
        "cafeteria",
        "healthcare",
        "club",
        "society",
        "phd",
        "mphil",
        "ms",
        "dpt",
        "diploma",
        "bba",
        "llb",
        "bs",
        "degree",
        "semester",
        "schedule",
        "timetable",
        "syllabus",
        "teacher",
        "professor"
    ],

    "ajku_complaint": [
        "complaint",
        "contact",
        "phone",
        "email",
        "dsa",
        "grievance",
        "code of conduct",
        "conduct",
        "policy",
        "rules",
        "discipline",
        "harassment",
        "report",
        "issue"
    ]
}

# ─────────────────────────────────────────────
# COLLECTION ROUTING
# ─────────────────────────────────────────────

def get_collection_for_query(query: str) -> str | None:

    q = query.lower()

    scores = {
        name: 0
        for name in COLLECTION_KEYWORDS
    }

    for name, keywords in COLLECTION_KEYWORDS.items():

        for kw in keywords:

            if kw in q:
                scores[name] += 1

    best = max(scores, key=scores.get)

    return best if scores[best] > 0 else None


def extract_question(text: str) -> str:
    match = re.search(r"Question:\s*(.+)", text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"^\d+\.\s*(.+)", text.strip())
    return match.group(1).strip() if match else text.strip().splitlines()[0][:120]


def has_urdu_script(text: str) -> bool:
    return any("\u0600" <= char <= "\u06ff" for char in text)

# ─────────────────────────────────────────────
# LOAD ALL DOCS FOR BM25
# ─────────────────────────────────────────────

def get_all_docs_for_bm25(collection):

    result = collection.get(
        include=["documents", "metadatas"]
    )

    docs = []

    for i, text in enumerate(result["documents"]):

        docs.append({
            "id": result["ids"][i],
            "text": text,
            "metadata": result["metadatas"][i]
        })

    return docs


def get_cached_docs(collection_name: str, collection) -> list[dict]:
    if collection_name not in docs_cache:
        docs = get_all_docs_for_bm25(collection)
        for doc in docs:
            metadata = doc.get("metadata") or {}
            metadata.setdefault("collection", collection_name)
            metadata.setdefault("question", extract_question(doc.get("text", "")))
            doc["metadata"] = metadata
        docs_cache[collection_name] = docs
    return docs_cache[collection_name]


def clear_retrieval_caches():
    global vocab_cache, question_cache
    docs_cache.clear()
    vocab_cache = None
    question_cache = None


def get_all_cached_questions() -> list[dict]:
    global question_cache

    if question_cache is not None:
        return question_cache

    questions = []
    for collection_name in COLLECTION_KEYWORDS:
        try:
            collection = chroma_client.get_collection(
                name=collection_name,
                embedding_function=embedding_fn
            )
            for doc in get_cached_docs(collection_name, collection):
                questions.append({
                    "question": extract_question(doc["text"]),
                    "collection": collection_name,
                    "doc_id": doc["id"],
                })
        except Exception:
            continue

    question_cache = questions
    return questions


def build_spell_vocab() -> set[str]:
    global vocab_cache

    if vocab_cache is not None:
        return vocab_cache

    words = set()
    for keywords in COLLECTION_KEYWORDS.values():
        for keyword in keywords:
            words.update(re.findall(r"[a-zA-Z]{3,}", keyword.lower()))

    for item in get_all_cached_questions():
        words.update(re.findall(r"[a-zA-Z]{3,}", item["question"].lower()))

    vocab_cache = words
    return words


def correct_query(query: str) -> tuple[str, bool]:
    if has_urdu_script(query):
        return query, False

    vocab = build_spell_vocab()
    corrected_tokens = []
    changed = False

    for token in re.findall(r"\w+|[^\w\s]+|\s+", query):
        if not token.isalpha() or len(token) < 4:
            corrected_tokens.append(token)
            continue

        lower = token.lower()
        if lower in vocab:
            corrected_tokens.append(token)
            continue

        match = difflib.get_close_matches(lower, vocab, n=1, cutoff=0.86)
        if match:
            replacement = match[0]
            corrected_tokens.append(replacement.capitalize() if token.istitle() else replacement)
            changed = True
        else:
            corrected_tokens.append(token)

    corrected = "".join(corrected_tokens)
    return corrected, changed and corrected.lower() != query.lower()

# ─────────────────────────────────────────────
# HYBRID SEARCH + RERANK RETRIEVAL
# ─────────────────────────────────────────────

def retrieve_documents(
    query: str,
    collection_name: str,
    n_results: int = 5
) -> list[dict]:

    try:

        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=embedding_fn
        )

    except Exception:
        return []

    all_docs_cache = get_cached_docs(collection_name, collection)

    # HYBRID SEARCH
    return hybrid_rerank_search(
        query=query,
        chroma_collection=collection,
        all_docs=all_docs_cache,
        top_k=10,
        top_n=n_results
    )


def retrieve_documents_federated(
    query: str,
    n_results: int = 5,
    per_collection_k: int = 8
) -> list[dict]:
    def search_collection(collection_name: str) -> list[dict]:
        try:
            collection = chroma_client.get_collection(
                name=collection_name,
                embedding_function=embedding_fn
            )
            all_docs_cache = get_cached_docs(collection_name, collection)
            candidates = hybrid_search(
                query=query,
                chroma_collection=collection,
                all_docs=all_docs_cache,
                top_k=per_collection_k,
            )
            for doc in candidates:
                metadata = doc.get("metadata") or {}
                metadata.setdefault("collection", collection_name)
                metadata.setdefault("question", extract_question(doc.get("text", "")))
                doc["metadata"] = metadata
            return candidates
        except Exception as exc:
            print(f"Federated search skipped {collection_name}: {exc}")
            return []

    candidates = []
    with ThreadPoolExecutor(max_workers=len(COLLECTION_KEYWORDS)) as executor:
        futures = {
            executor.submit(search_collection, collection_name): collection_name
            for collection_name in COLLECTION_KEYWORDS
        }
        for future in as_completed(futures):
            candidates.extend(future.result())

    deduped: dict[str, dict] = {}
    for doc in candidates:
        key = doc.get("id")
        if not key:
            # Assign a fallback key so docs without an id are not all collapsed
            key = f"_noid_{len(deduped)}"
            doc["id"] = key
        if key not in deduped:
            deduped[key] = doc
        else:
            # Merge score fields from duplicate hits (e.g. same doc found by both BM25 & vector)
            deduped[key].update({k: v for k, v in doc.items() if k.endswith("_score")})
            # Keep the richer metadata (longer source value)
            existing_meta = deduped[key].get("metadata") or {}
            new_meta = doc.get("metadata") or {}
            merged_meta = {**existing_meta, **{k: v for k, v in new_meta.items() if v}}
            deduped[key]["metadata"] = merged_meta

    # Ensure every doc has collection in metadata before reranking
    for doc in deduped.values():
        meta = doc.get("metadata") or {}
        if not meta.get("collection"):
            meta["collection"] = "unknown"
        if not meta.get("question"):
            meta["question"] = extract_question(doc.get("text", ""))
        doc["metadata"] = meta

    return rerank(query, list(deduped.values()), top_n=n_results)


def retrieve_chunks(
    query: str,
    collection_name: str | None = None,
    n_results: int = 5
) -> list[str]:

    top_docs = (
        retrieve_documents(query, collection_name, n_results)
        if collection_name
        else retrieve_documents_federated(query, n_results)
    )

    # RETURN ONLY TEXT
    context_chunks = [
        doc["text"]
        for doc in top_docs
    ]

    return context_chunks


def build_sources(docs: list[dict]) -> list[dict]:
    sources = []
    seen = set()

    for index, doc in enumerate(docs, start=1):
        metadata = doc.get("metadata") or {}
        title = metadata.get("question") or extract_question(doc.get("text", ""))
        source = metadata.get("source") or metadata.get("file") or doc.get("id", "")
        key = (title, source)

        if key in seen:
            continue
        seen.add(key)

        sources.append({
            "label": f"S{index}",
            "title": title,
            "source": source,
            "doc_id": doc.get("id", ""),
            "score": round(float(doc.get("rerank_score", 0.0)), 3),
        })

    return sources


def confidence_from_docs(docs: list[dict]) -> int:
    if not docs:
        return 0

    top = float(docs[0].get("rerank_score", 0.0))
    method = docs[0].get("rerank_method", "")
    if method == "cross_encoder":
        confidence = 1 / (1 + math.exp(-top))
    else:
        confidence = min(1.0, top / 8)

    if len(docs) >= 2:
        gap = abs(top - float(docs[1].get("rerank_score", 0.0)))
        confidence = min(1.0, confidence + min(gap / 10, 0.15))

    return int(round(max(0.0, min(confidence, 1.0)) * 100))


def generate_follow_up_suggestions(query: str, docs: list[dict], limit: int = 3) -> list[str]:
    suggestions = []
    current = query.lower().strip()

    for doc in docs:
        question = extract_question(doc.get("text", ""))
        if question and question.lower().strip() != current and question not in suggestions:
            suggestions.append(question)
        if len(suggestions) >= limit:
            return suggestions

    collection = docs[0].get("metadata", {}).get("collection") if docs else None
    for item in get_all_cached_questions():
        if collection and item["collection"] != collection:
            continue
        question = item["question"]
        if question.lower().strip() != current and question not in suggestions:
            suggestions.append(question)
        if len(suggestions) >= limit:
            break

    return suggestions


def get_quick_questions(limit: int = 3) -> list[str]:
    questions = []
    seen_collections = set()

    for item in get_all_cached_questions():
        question = item["question"]
        if item["collection"] not in seen_collections and question not in questions:
            questions.append(question)
            seen_collections.add(item["collection"])
        if len(questions) >= limit:
            break

    if len(questions) < limit:
        for item in get_all_cached_questions():
            if item["question"] not in questions:
                questions.append(item["question"])
            if len(questions) >= limit:
                break

    return questions

# ─────────────────────────────────────────────
# GEMINI ANSWER GENERATION
# ─────────────────────────────────────────────

def generate_answer(
    query: str,
    context_chunks: list[str],
    history: str = ""
) -> str:

    if not context_chunks:

        return (
            "Mujhe is sawaal ka jawab available data mein nahi mila. "
            "Please AJKU ke official website https://ajku.edu.pk "
            "ya relevant department se rabta karein."
        )

    context = "\n\n".join(context_chunks)

    history_section = ""

    if history:

        history_section = f"""

Pichli conversation:
{history}

"""

    prompt = f"""
Tum AJKU (Azad Jammu & Kashmir University)
ke ek helpful aur knowledgeable assistant ho.

Tumhara kaam students ke sawaalon ka jawab dena hai
sirf neeche diye gaye context ke basis par.

Context:
{context}

{history_section}

Student Question:
{query}

Instructions:

1. Sirf context ka use karo
2. Friendly aur clear answer do
3. Agar answer context mein nahi hai toh official source refer karo
4. User jis language/script mein pooche usi mein jawab do: English, Urdu, Roman Urdu, ya mixed
5. Bullet points use karo agar multiple points hoon
6. Short aur useful answer do
7. Important facts ke baad source markers use karo jaise [S1], [S2]

Answer:
"""

    try:

        last_error = None
        client = get_gemini_client()
        for model_name in get_valid_gemini_models():
            for attempt in range(GEMINI_MAX_RETRIES):
                try:
                    chat = client.chats.create(model=model_name)
                    response = chat.send_message(prompt)
                    text = (response.text or "").strip()
                    if text:
                        return text
                    last_error = RuntimeError(f"Empty Gemini response from {model_name}")
                except Exception as model_error:
                    last_error = model_error
                    print(
                        f"Gemini model failed ({model_name}, attempt {attempt + 1}/"
                        f"{GEMINI_MAX_RETRIES}): {summarize_gemini_error(model_error)}"
                    )
                    if not is_retryable_gemini_error(model_error):
                        break
                    if attempt < GEMINI_MAX_RETRIES - 1:
                        sleep_before_retry(attempt)

        raise last_error

    except Exception as e:

        print("Gemini Error:", summarize_gemini_error(e))

        return context_fallback_answer(context_chunks)


def stream_answer(
    query: str,
    context_chunks: list[str],
    history: str = ""
):
    if not context_chunks:
        yield (
            "Mujhe is sawaal ka jawab available data mein nahi mila. "
            "Please AJKU ke official website https://ajku.edu.pk "
            "ya relevant department se rabta karein."
        )
        return

    context = "\n\n".join(context_chunks)
    history_section = f"\n\nPichli conversation:\n{history}\n" if history else ""
    prompt = f"""
Tum AJKU (Azad Jammu & Kashmir University) ke helpful assistant ho.
Sirf diye gaye context ke basis par jawab do.

Context:
{context}

{history_section}

Student Question:
{query}

Instructions:
1. User jis language/script mein pooche usi mein jawab do: English, Urdu, Roman Urdu, ya mixed
2. Important facts ke baad source markers use karo jaise [S1], [S2]
3. Agar answer context mein nahi hai toh official source refer karo
4. Short, clear, friendly answer do

Answer:
"""

    try:
        last_error = None
        client = get_gemini_client()
        for model_name in get_valid_gemini_models():
            for attempt in range(GEMINI_MAX_RETRIES):
                try:
                    chat = client.chats.create(model=model_name)
                    response = chat.send_message_stream(prompt)
                    yielded = False
                    for chunk in response:
                        text = getattr(chunk, "text", "")
                        if text:
                            yielded = True
                            yield text
                    if yielded:
                        return
                    last_error = RuntimeError(f"Empty Gemini stream from {model_name}")
                except Exception as model_error:
                    last_error = model_error
                    print(
                        f"Gemini stream model failed ({model_name}, attempt {attempt + 1}/"
                        f"{GEMINI_MAX_RETRIES}): {summarize_gemini_error(model_error)}"
                    )
                    if not is_retryable_gemini_error(model_error):
                        break
                    if attempt < GEMINI_MAX_RETRIES - 1:
                        sleep_before_retry(attempt)

        raise last_error
    except Exception as e:
        print("Gemini Stream Error:", summarize_gemini_error(e))
        yield generate_answer(query, context_chunks, history)

# ─────────────────────────────────────────────
# MAIN QUERY PROCESSOR
# ─────────────────────────────────────────────

def process_query(
    query: str,
    user_id: int = 0,
    history: str = ""
) -> dict:

    corrected_query, was_corrected = correct_query(query)

    docs = retrieve_documents_federated(
        corrected_query,
        n_results=5
    )
    chunks = [doc["text"] for doc in docs]
    confidence_score = confidence_from_docs(docs)
    top_doc = docs[0] if docs else {}
    collection_name = top_doc.get("metadata", {}).get("collection", "none")

    if confidence_score < MIN_CONFIDENCE_SCORE:
        answer = LOW_CONFIDENCE_MESSAGE
    else:
        answer = generate_answer(
            corrected_query,
            chunks,
            history
        )


    return {
        "answer": answer,
        "collection": collection_name,
        "chunks_found": len(chunks),
        "retrieved_docs": docs,
        "top_rerank_score": top_doc.get("rerank_score", 0.0),
        "rerank_method": top_doc.get("rerank_method", ""),
        "corrected_query": corrected_query,
        "was_corrected": was_corrected,
        "confidence_score": confidence_score,
        "sources": build_sources(docs),
        "follow_up_suggestions": generate_follow_up_suggestions(corrected_query, docs, 3),
    }


def prepare_query(
    query: str,
    history: str = ""
) -> dict:
    corrected_query, was_corrected = correct_query(query)
    docs = retrieve_documents_federated(corrected_query, n_results=5)
    top_doc = docs[0] if docs else {}
    collection_name = top_doc.get("metadata", {}).get("collection", "none")
    return {
        "query": query,
        "corrected_query": corrected_query,
        "was_corrected": was_corrected,
        "collection": collection_name,
        "chunks": [doc["text"] for doc in docs],
        "docs": docs,
        "sources": build_sources(docs),
        "confidence_score": confidence_from_docs(docs),
        "follow_up_suggestions": generate_follow_up_suggestions(corrected_query, docs, 3),
        "history": history,
    }

# ─────────────────────────────────────────────
# QUERY VALIDATION
# ─────────────────────────────────────────────

BLOCKED_WORDS = [
    "politics",
    "election",
    "violence",
    "abuse",
    "hate",
    "sex",
    "drug",
    "kill",
    "terror",
    "weapon"
]

def is_valid_query(query: str):

    q = query.lower()

    for word in BLOCKED_WORDS:

        if re.search(rf"\b{re.escape(word)}\b", q):
            return False, "Inappropriate content detected."

    if len(query.strip()) < 3:
        return False, "Query too short."

    return True, "OK"

# ─────────────────────────────────────────────
# SEMANTIC ROUTE LABEL
# ─────────────────────────────────────────────

def semantic_route(query: str) -> str:

    name = get_collection_for_query(query)

    route_map = {
        "ajku_admission": "admission",
        "ajku_scholarship": "scholarship",
        "ajku_exam": "exam",
        "ajku_academic": "academic",
        "ajku_complaint": "complaint",
    }

    return route_map.get(name, "general")

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# CLI TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print(
        "AJKU AI Assistant. Type 'exit' to quit.\n"
    )

    hist = ""

    while True:

        q = input("You: ").strip()

        if q.lower() == "exit":
            break

        valid, reason = is_valid_query(q)

        if not valid:

            print(f"Bot: {reason}\n")

            continue

        result = process_query(
            q,
            history=hist
        )

        print(
            f"Bot [{result['collection']}]: "
            f"{result['answer']}\n"
        )

        hist += (
            f"User: {q}\n"
            f"Assistant: {result['answer']}\n"
        )
