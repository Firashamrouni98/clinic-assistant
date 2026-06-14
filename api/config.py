from dotenv import load_dotenv
import os

load_dotenv()

# ── Groq ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ── Clinic info (change this per client) ──────────────────────────────────────
CLINIC_NAME     = "Clinique Al Amal"
CLINIC_LOCATION = "Ariana, Tunis"
CLINIC_PHONE    = "+216 71 000 000"
CLINIC_HOURS    = "Lundi–Samedi 8h–19h"
CLINIC_SERVICES = ["Médecine générale", "Neurologie", "Ophtalmologie", "Dentisterie"]
CLINIC_FEES     = {"generale": "50 TND", "specialiste": "80–120 TND"}
CLINIC_EMAIL    = "contact@amal-clinic.tn"
EMERGENCY_NUM   = "190"

# ── Milvus RAG ────────────────────────────────────────────────────────────────
MILVUS_URI      = "./milvus_clinic.db"
COLLECTION_NAME = "clinic_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384
TOP_K_RESULTS   = 3