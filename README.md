# Clinic AI Assistant

A production-ready RAG-powered AI receptionist for medical clinics, built to automate appointment booking, FAQ handling, and patient triage in French, English, and Arabic. Deployed live on Railway, demoed to private clinics in Ariana, Tunis.

**Live demo:** [web-production-808b2.up.railway.app/docs](https://web-production-808b2.up.railway.app/docs)

---

## Why this exists

Private clinics in Tunisia handle a large volume of repetitive phone calls: hours, pricing, insurance questions, and appointment scheduling. This assistant absorbs that load 24/7, in the patient's own language, while staying strictly within safe boundaries (no diagnoses, immediate redirect to emergency services when needed).

This project was built as a freelance product to demo directly to clinic owners, not a tutorial exercise. The architecture reflects that: it had to actually run reliably for a non-technical clinic staff member to use the same day.

---

## What it does

- Answers patient questions about hours, pricing, services, insurance (CNAM), first-visit requirements, cancellation policy, and more, using a 40-entry knowledge base specific to the clinic
- Detects the patient's language automatically (French, English, or Arabic) and responds in kind, without language drift
- Collects appointment requests step by step (name, then specialty, then availability, then phone), instead of overwhelming the patient with a single long form
- Immediately redirects any mention of a medical emergency to the national emergency number, without attempting to continue the conversation
- Never provides a medical diagnosis, by design and by explicit system instruction

---

## Architecture

```
Patient message
      │
      ▼
FastAPI endpoint (/chat)
      │
      ▼
Embed query (sentence-transformers, all-MiniLM-L6-v2)
      │
      ▼
Semantic search (Milvus Lite, local vector store)
      │
      ▼
Inject retrieved context into system prompt
      │
      ▼
Groq (llama-3.3-70b-versatile) generates response
      │
      ▼
JSON response back to client
```

**Stack:**
- **API:** FastAPI + Uvicorn
- **LLM:** Groq (llama-3.3-70b-versatile), via the official Groq SDK
- **Vector store:** Milvus Lite (embedded, file-based, no external DB server needed)
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`, 384 dimensions)
- **Frontend:** Vanilla HTML/CSS/JS chat widget, no framework dependency
- **Deployment:** Railway, Python 3.11 (pinned — see note below)

---

## Project structure

```
clinic-assistant/
├── api/
│   ├── main.py            FastAPI app, /chat and / endpoints
│   └── config.py          Clinic identity, model config, constants
├── agent/
│   └── clinic_agent.py    System prompt, Groq call, language/safety rules
├── rag/
│   ├── loader.py          Embeds clinic_faq.txt into Milvus
│   └── retriever.py       Semantic search against the vector store
├── data/
│   └── clinic_faq.txt     Knowledge base (40+ Q&A entries, French)
├── chat/
│   └── index.html         Standalone chat widget, connects to the API
├── milvus_clinic.db/      Pre-built vector store (committed, ~50KB)
├── requirements.txt
├── runtime.txt             Pins Python 3.11 for Railway build compatibility
├── Procfile                Railway start command
└── .env.example
```

---

## Running locally

```bash
python -m venv venv
.\venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Set your Groq API key
cp .env.example .env             # then edit .env with your key

# Build the knowledge base (only needed once, or after editing clinic_faq.txt)
python rag/loader.py

# Start the API
uvicorn api.main:app --reload --port 8000
```

Open `chat/index.html` in a browser to use the widget against your local API. Full interactive API docs are available at `http://localhost:8000/docs`.

---

## Deployment notes

Deployed on Railway. Two non-obvious fixes were required to get there, documented here in case they save someone else the same debugging cycle:

1. **Python version pinning.** Railway's default builder (Railpack) picks Python 3.13 unless told otherwise. `pydantic-core==2.18.4`'s Rust bindings (PyO3 0.21.2) only support up to Python 3.12, so the build fails trying to compile from source. Fixed with a `runtime.txt` containing `python-3.11`.
2. **Trimmed dependencies.** The original `requirements.txt` carried dead weight from early experimentation (LangChain, LangGraph, ChromaDB) that were never actually imported anywhere in the codebase. Trimming to only the packages actually used in the import graph cut the dependency tree significantly and avoided unnecessary build size and time.

The Milvus Lite vector store is small enough (~50KB for 40 FAQ entries) to commit directly to the repo, so the knowledge base is ready immediately on deploy with no build-time indexing step.

---

## For clinics: what this means for you

If you're a clinic owner evaluating this: this assistant answers the questions your front desk gets asked dozens of times a day, in whichever language the patient prefers, without the patient waiting on hold. It never guesses at medical advice and routes any urgent-sounding message straight to the emergency line. Appointment requests are captured cleanly (name, specialty, availability, phone) and can be handed to your staff to confirm.

It runs on your own branded chat widget, embeddable on your clinic's website, and the knowledge base is fully customizable to your actual hours, pricing, and policies.

---

## Author

Built by Firas Hamrouni, Telecommunications Engineering graduate (ESPRIT Tunis), AI engineer focused on LLMs, RAG pipelines, and multi-agent systems.