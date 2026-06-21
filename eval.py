import json
from generate_answer import generate_answer

with open("eval_set.json") as f:
    eval_data = json.load(f)

results = []

for i, item in enumerate(eval_data):
    question = item["question"]
    expected = item["expected_answer"]

    answer, used_chunks = generate_answer(question)

    print(f"\n{'='*60}")
    print(f"Q{i+1}: {question}")
    print(f"EXPECTED: {expected}")
    print(f"GOT:      {answer}")

    score = input("Score this answer (c = correct, p = partial, i = incorrect): ").strip().lower()
    results.append({"question": question, "score": score})

correct = sum(1 for r in results if r["score"] == "c")
partial = sum(1 for r in results if r["score"] == "p")
incorrect = sum(1 for r in results if r["score"] == "i")
total = len(results)

print(f"\n{'='*60}")
print(f"EVAL RESULTS: {correct}/{total} correct, {partial}/{total} partial, {incorrect}/{total} incorrect")
print(f"Accuracy (correct only): {correct/total*100:.1f}%")