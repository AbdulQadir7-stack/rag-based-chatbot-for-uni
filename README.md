# AJKU AI Assistant 🎓

AI-powered multi-agent chatbot for University of Azad Jammu & Kashmir.
Built with Gemini AI + ChromaDB RAG + FastAPI + Streamlit.

---

## Folder Structure

```
ajku_project/
│
├── data/                          ← Put your FAQ .txt files HERE
│   ├── admission fee faqs.txt
│   ├── academic faqs.txt
│   ├── scholarships faqs.txt
│   ├── exam faqs.txt
│   └── complaint faqs.txt
│
├── chroma_db_lite/                ← Auto-created by ChromaDB
│
├── agents.py                      ← Multi-agent RAG logic + Gemini AI
├── api.py                         ← FastAPI backend (JWT auth, chat, admin)
├── app.py                         ← Streamlit frontend (chat UI + dashboard)
├── database.py                    ← SQLite database layer
├── populate_multicol.py           ← Load FAQ data into ChromaDB
├── debug_collections.py           ← Inspect ChromaDB data
├── test_retrieval.py              ← Test vector search
├── requirements.txt               ← Python dependencies
└── .env                           ← API keys (keep private!)
```

---

## Setup Steps (Step by Step)

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Set up .env file
Create `.env` in the project folder:
```
GOOGLE_API_KEY=your_google_api_key_here
JWT_SECRET_KEY=replace_with_a_long_random_secret
GEMINI_MODEL=gemini-3.6-flash
```
Get your key from: https://aistudio.google.com/app/apikey

### Step 3: Add your FAQ data
Create a `data/` folder and add these files:
- `admission fee faqs.txt`
- `academic faqs.txt`
- `scholarships faqs.txt`
- `exam faqs.txt`
- `complaint faqs.txt`

Each file format:
```
Question: Admission fee kitni hai?
Answer: Admission form ki fee 2450 rupees hai...

Question: Last date kab hai?
Answer: Admission ki last date August hai...
```

### Step 4: Populate ChromaDB
```bash
python populate_multicol.py
```

### Step 5: Start the backend API
```bash
python api.py
```
Backend runs at: http://localhost:8000

### Step 6: Start the Streamlit frontend (new terminal)
```bash
streamlit run app.py
```
Frontend opens at: http://localhost:8501

---

## Default Login Credentials

| Username | Password   | Role    |
|----------|------------|---------|
| admin    | admin123   | Admin   |
| student  | student123 | Student |

---

## Features

- ✅ Multi-agent routing (5 specialized agents)
- ✅ Gemini AI generates natural answers from FAQ context
- ✅ JWT authentication (login/logout)
- ✅ Admin dashboard with charts
- ✅ Cross-Encoder reranking for better retrieval precision
- ✅ Server-side conversation memory for follow-up questions
- ✅ Evaluation framework: HitRate@1, MRR, optional BERTScore
- ✅ Admin analytics: trends, active users, collections, reranker usage
- ✅ Query spell correction for English/Roman Urdu terms
- ✅ Multilingual answers in English, Urdu, Roman Urdu, or mixed language
- ✅ Source citations, confidence score, streaming responses, and follow-up suggestions
- ✅ Data-driven quick questions from the FAQ dataset
- ✅ Multi-collection federated search across all 5 Chroma collections
- ✅ Confidence-gated hallucination prevention for low-relevance queries
- ✅ Admin-only retrieval debug info and student-clean responses
- ✅ Rate limiting on chat endpoints with SlowAPI
- ✅ Admin retrieval-cache reload endpoint
- ✅ Query logging
- ✅ Thumbs up/down feedback
- ✅ Chat history context (last 4 exchanges)
- ✅ Quick suggestion buttons
- ✅ User management (admin can create users)
- ✅ Inappropriate query filtering

---

## Troubleshooting

**"Backend not running"** → Run `python api.py` first

**"Collection not found"** → Run `python populate_multicol.py`

**"Invalid API key"** → Check your `.env` file has correct GOOGLE_API_KEY

**Debug ChromaDB**:
```bash
python debug_collections.py
python test_retrieval.py
```

## Evaluation

Create an evaluation dataset from `eval_dataset.example.json`, then run:

```bash
python evaluation.py --dataset eval_dataset.json --top-k 5
```

Latest federated retrieval run:

| Metric | Value |
|---|---:|
| HitRate@1 | 0.9831 |
| MRR | 0.9831 |
| Count | 59 |

| Collection | HitRate@1 | MRR | Count |
|---|---:|---:|---:|
| ajku_academic | 1.0 | 1.0 | 12 |
| ajku_admission | 1.0 | 1.0 | 15 |
| ajku_complaint | 1.0 | 1.0 | 7 |
| ajku_exam | 1.0 | 1.0 | 12 |
| ajku_scholarship | 0.9231 | 0.9231 | 13 |

For answer-quality scoring with BERTScore:

```bash
python evaluation.py --dataset eval_dataset.json --top-k 5 --bertscore
```
