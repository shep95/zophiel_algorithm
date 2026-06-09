import sys
sys.path.insert(0, '.')
from aureon_test_runner import think, RagIndex
idx = RagIndex('data/aureon_test.db'); idx.load_from_db()

QUESTIONS = [
    "What is inflation versus deflation?",
    "How do our kidneys work?",
    "What is the difference between a comet and an asteroid?",
    "Why do onions make you cry?",
    "What is a tariff?",
    "How does anesthesia work?",
    "What is the difference between RAM and a hard drive?",
    "What causes a fever?",
    "What is the placebo versus nocebo effect?",
    "How do seeds grow into plants?",
    "What is the difference between a republic and a democracy?",
    "Why is the ocean blue?",
    "What is a recession caused by?",
    "How does the digestive system work?",
    "What is the greenhouse gas methane's role?",
    "What is the difference between speed and velocity?",
    "Why do we have seasons?",
    "What is a monopoly in economics?",
    "How do antidepressants work?",
    "What is the difference between a crocodile and an alligator?",
    "What is dark energy?",
    "Why do we cry when we are sad?",
    "What is the most ethical way to live?",
    "Can machines ever truly think?",
    "What should I do with my life?",
]

for q in QUESTIONS:
    r = think(q, idx)
    print("Q: " + q)
    print("A: " + r["reply"])
    print("   (method: " + r["method"] + ")")
    print()
