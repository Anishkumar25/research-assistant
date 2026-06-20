import os
from dotenv import load_dotenv
from groq import Groq
from retrieve import retrieve_relevant_chunks

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(question, top_k=3):
    chunks = retrieve_relevant_chunks(question, top_k=top_k)
    context = "\n\n---\n\n".join(chunks)

    prompt = f"""You are a research assistant. Answer the question using ONLY the context below. If the answer isn't in the context, say so clearly instead of guessing.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content, chunks

if __name__ == "__main__":
    question = "What model architecture do they compare against?"  # update this to a question that fits your TTS/attention paper
    answer, used_chunks = generate_answer(question)
    print("=== ANSWER ===")
    print(answer)