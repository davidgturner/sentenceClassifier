import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example sentences for classification
sentences = [
    "What is the capital of France?",
    "I love to go for long walks in the park.",
    "How many planets are there in our solar system?",
    "The cat is sitting on the mat.",
    "Who won the World Series in 2020?"
]

# Tokenize the input sentences
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

# Perform the classification
outputs = model(**inputs)
logits = outputs.logits
predicted_classes = torch.argmax(logits, dim=1).tolist()

# Get the predicted class labels
class_labels = model.config.id2label
predicted_labels = [class_labels[pred_class] for pred_class in predicted_classes]

# Print the predicted class labels for each sentence
for sentence, predicted_label in zip(sentences, predicted_labels):
    print(f"Sentence: {sentence}")
    print(f"Predicted class: {predicted_label}")
    print()
