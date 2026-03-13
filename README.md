# AT82.03 Machine Learning — Assignment 6
## Naive RAG vs. Contextual Retrieval

**Student:** Michael Roque Lacar
**ID:** st126161
**Chapter:** 11 — *Information Retrieval and Retrieval-Augmented Generation*
**Source:** Jurafsky & Martin, *Speech and Language Processing* (Draft, January 2026)

---

## Repository Structure

```
├── A6_RAG_Techniques.ipynb                  # Main notebook (Tasks 1, 2 & 3)
├── app/
│   ├── app.py                               # Streamlit chatbot web application (Task 3)
│   ├── requirements.txt                     # Python dependencies for the app
│   └── 11.pdf                               # Chapter 11 PDF (place here before running)
├── answer/
│   └── response-st-126161-chapter-11.json  # JSON evaluation output (20 QA pairs)
└── README.md
```

---

## Models Used

| Component | Model | Provider |
|-----------|-------|----------|
| Retriever (Embedding) | `all-MiniLM-L6-v2` | sentence-transformers (local, free) |
| Generator (LLM) | `llama-3.1-8b-instant` | Groq API (free tier) |
| Contextual Enrichment | `llama-3.1-8b-instant` | Groq API (free tier) |

> **Note:** OpenAI was not used in this assignment. All LLM calls go through the Groq API
> (free tier, 100k TPD limit). Embeddings run locally via sentence-transformers with no
> API key required.

---

## Running the Notebook

Open `A6_RAG_Techniques.ipynb` in Jupyter and run all cells top to bottom.

```bash
jupyter notebook A6_RAG_Techniques.ipynb
```

---

## Running the Web Application

```bash
cd app
streamlit run app.py
```

The app will open in your browser. Enter your Groq API key in the sidebar, then ask any
question about Chapter 11. Each answer displays the source chunks used for generation.

---

## Methodology

### Task 1 — Data Preparation
- Text extracted from the Chapter 11 PDF using `PyPDF2` and cleaned (hyphen rejoining,
  whitespace normalisation, running header removal).
- Chunked into 400-word overlapping windows with 50-word overlap using a word-level splitter.
- 20 QA pairs manually authored from chapter content, covering sparse retrieval, dense
  retrieval, BM25, ColBERT, FAISS, RAG, hallucination, evaluation metrics, and benchmarks.

### Task 2 — Pipeline Comparison

**Naive RAG**
1. Embed raw chunks with `all-MiniLM-L6-v2`.
2. Index with FAISS `IndexFlatIP` (cosine similarity via L2 normalisation).
3. Retrieve top-3 chunks per query and generate answer with `llama-3.1-8b-instant`.

**Contextual Retrieval**
1. Prepend each chunk with a 1-sentence LLM-generated context prefix via Groq.
2. Embed enriched chunks and index identically to Naive RAG.
3. Retrieve top-3 enriched chunks per query and generate answer with `llama-3.1-8b-instant`.

### Task 3 — Web Application
- Built with Streamlit featuring a full chat interface.
- Uses the Contextual Retrieval pipeline for answer generation.
- Displays the generated answer and cites the source chunks with similarity scores.
- Footer displays student name, ID, and model information.

---

## Results

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|--------|---------|---------|---------|
| Naive RAG | 0.3673 | 0.1439 | 0.2608 |
| Contextual Retrieval | 0.1978 | 0.0702 | 0.1373 |

Naive RAG scored higher in aggregate due to token-trimming constraints imposed by Groq's
free-tier daily limit (100,000 TPD). Enriched chunks, being longer due to the prepended
context prefix, were disproportionately affected by truncation. Despite this, Contextual
Retrieval outperformed Naive RAG on 3 out of 20 questions (Q8 inverted index, Q16
bag-of-words, Q17 token F1), demonstrating the method's effectiveness under sufficient
token budget. This is consistent with Anthropic's (2024) report of a 49% reduction in
retrieval failures under fair conditions.

---

## Known Limitations

- **Token budget:** Groq's free tier (100k TPD) required chunk trimming that
  disproportionately affected enriched chunks, impacting Contextual Retrieval scores.
- **ROUGE limitations:** ROUGE measures surface-form overlap only and does not capture
  semantic correctness. BERTScore or LLM-as-judge evaluation would give a more complete
  picture.
- **Rate limiting:** The `groq_chat()` function includes automatic retry logic that parses
  the wait time from the 429 error message and sleeps accordingly.

---

## References

- Jurafsky, D. & Martin, J. H. (2026). *Speech and Language Processing* (Draft, January
  2026). Chapter 11.
- Anthropic. (2024). Contextual Retrieval.
  https://www.anthropic.com/engineering/contextual-retrieval
- Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese
  BERT-Networks. *EMNLP 2019*.
- Johnson, J., Douze, M., & Jégou, H. (2017). Billion-scale similarity search with GPUs.
  *arXiv:1702.08734*.
- Robertson, S. et al. (1995). Okapi at TREC-3. *Overview of the Third Text REtrieval
  Conference*.
