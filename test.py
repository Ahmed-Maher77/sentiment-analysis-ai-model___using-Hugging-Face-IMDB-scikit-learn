from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="./sentiment-model"
)

examples = [
    "This movie was absolutely amazing!",
    "I loved every minute of it.",
    "The film was terrible and boring.",
    "I would never watch this again."
]

for text in examples:
    result = classifier(text)
    print(text)
    print(result)
    print("-" * 50)