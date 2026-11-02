from llm import generate_answer


question = "What is machine learning?"


context = """
Machine learning is a branch of artificial
intelligence that allows computers to learn
patterns from data and make predictions or
decisions without being explicitly programmed
for every task.
"""


answer = generate_answer(
    question,
    context
)


print("\nQuestion:")
print(question)

print("\nAnswer:")
print(answer)