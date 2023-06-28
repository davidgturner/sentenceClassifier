# sentenceClassifier
classifies sentences into questions or statements

To setup:
``` pip install torch transformers ```

BERT models:

bert-base-uncased: Base BERT model with uncased vocabulary.
bert-base-cased: Base BERT model with cased vocabulary.
bert-large-uncased: Large BERT model with uncased vocabulary.
bert-large-cased: Large BERT model with cased vocabulary.
RoBERTa models:

roberta-base: Base RoBERTa model.
roberta-large: Large RoBERTa model.
GPT models:

gpt2: OpenAI's GPT-2 model.
DistilBERT models:

distilbert-base-uncased: Distilled version of BERT with uncased vocabulary.
distilbert-base-cased: Distilled version of BERT with cased vocabulary.
ALBERT models:

albert-base-v2: Base ALBERT model.
XLNet models:

xlnet-base-cased: Base XLNet model with cased vocabulary.
xlnet-large-cased: Large XLNet model with cased vocabulary.
Electra models:

google/electra-base-generator: Base Electra model.
T5 models:

t5-base: Base T5 model.

(https://huggingface.co/models) to discover more models

This also has a demo of using MLFlow too. 

1. ``` pip install mlflow ```
2. run ``` python3 create_experiment.py ``` - create an MLflow experiment named "dataset_versioning,"
3. run ``` python3 track_version.py ``` - Track and version dataset using MLflow
4. run ``` python3 create_new_version.py ``` - Update the dataset and create a new version
5. run ``` python3 view_version.py {version_name} ``` - View and retrieve dataset versions using MLflow
