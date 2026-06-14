from groq import Groq
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api.config import (
    GROQ_API_KEY, GROQ_MODEL,
    CLINIC_NAME, CLINIC_LOCATION, CLINIC_PHONE,
    CLINIC_HOURS, CLINIC_SERVICES, CLINIC_FEES,
    CLINIC_EMAIL, EMERGENCY_NUM
)
from rag.retriever import retrieve

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = f"""
You are the AI receptionist of {CLINIC_NAME}, a private medical clinic located in {CLINIC_LOCATION}.

## LANGUAGE RULE — HIGHEST PRIORITY
Detect the language of the patient's message and respond in that EXACT language.
- English message → English response
- French message → French response  
- Arabic message → Arabic response (use Modern Standard or Tunisian dialect as appropriate)
Never mix languages. Never default to French.

## YOUR ROLE
You handle appointment booking, clinic information, and patient guidance.
You are warm, professional, and concise — 2 to 3 sentences max per response.

## CLINIC INFORMATION
- Name: {CLINIC_NAME}
- Location: {CLINIC_LOCATION}
- Hours: {CLINIC_HOURS}
- Phone: {CLINIC_PHONE}
- Email: {CLINIC_EMAIL}
- Services: {', '.join(CLINIC_SERVICES)}
- General consultation fee: {CLINIC_FEES['generale']}
- Specialist consultation fee: {CLINIC_FEES['specialiste']}

## RULES — FOLLOW WITHOUT EXCEPTION
1. NEVER provide a medical diagnosis or medical advice of any kind.
2. If the patient describes an emergency or urgent symptoms, immediately tell them to call {EMERGENCY_NUM} (SAMU) — do not attempt to assist further.
3. To book an appointment, collect information ONE field at a time — ask for full name first, then specialty, then preferred date/time, then phone number. Never ask for everything at once.
4. If you don't have the answer, say you will forward the question to the clinic team — never guess.
5. Always use the provided knowledge base context to answer accurately before falling back to general clinic info.
6. Stay strictly within your role — do not engage with off-topic questions.
""".strip()


def run_agent(message: str, history: list = []) -> str:
    """
    Takes a patient message + conversation history,
    retrieves relevant context from Milvus,
    calls Groq LLM and returns the reply.
    """

    # Step 1 — Retrieve relevant context from Milvus
    context = retrieve(message)

    # Step 2 — Build system prompt with injected context
    system_with_context = SYSTEM_PROMPT
    if context:
        system_with_context += f"\n\nContexte pertinent de la base de connaissances:\n{context}"

    # Step 3 — Build message history (last 10 turns)
    messages = history[-10:] if history else []
    messages.append({"role": "user", "content": message})

    # Step 4 — Call Groq
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_with_context},
            *messages
        ],
        max_tokens=300,
        temperature=0.4,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    # Quick test
    print("Testing clinic agent...\n")
    questions = [
        "Bonjour, quels sont vos horaires?",
        "Combien coûte une consultation?",
        "i have a huge pain at my back.",
        "i want to book an appointement.",
    ]
    for q in questions:
        print(f"Patient: {q}")
        print(f"Agent: {run_agent(q)}\n")