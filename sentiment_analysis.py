from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model = model_name, tokenizer = tokenizer)
results = classifier(["We are very happy to show you the modified transformers library.",
                    "We hope you don't hate it."])

for result in results:
    print(result)

tokens = tokenizer.tokenize("we are very happy to show you the tokenizer library.")
token_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = tokenizer("We are very happy to show you the transformers lib")

print("Tokens: ", tokens)
print("token_ids: ", token_ids)
print("input_ids: ", input_ids)