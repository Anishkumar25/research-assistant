import os
from dotenv import load_dotenv
from groq import Groq
from retrieve import retrieve_relevant_chunks

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(question, top_k=5):
    chunks = retrieve_relevant_chunks(question, top_k=top_k)
    context = "\n\n---\n\n".join(chunks)

    prompt = f"""You are a research assistant. Answer the question using ONLY the context below.

Rules:
- If the answer is clearly stated in the context, answer it directly and specifically.
- If the answer is NOT in the context, respond with exactly this sentence and nothing else: "The provided context does not contain this information."
- Never generate unrelated questions, quizzes, or content outside of directly answering the question asked.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=500,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content, chunks

if __name__ == "__main__":
    question = "How many ODE steps are used in the generation process?"
    answer, used_chunks = generate_answer(question)
    print("=== ANSWER ===")
    print(answer)