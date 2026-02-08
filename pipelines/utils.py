from zenml import pipeline
from steps.load_model import load_production_model
from zenml.client import Client
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
from mlflow.tracking import MlflowClient
from typing import Tuple
import pandas as pd

def get_data_for_test(subject:str, body:str):

    def clean(text):
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        # Corrected line:
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = text.split()
        text = [w for w in text if w.isalpha() and len(w) > 1]
        return " ".join(text)
    
    client = Client()

    tokenizer_artifact = client.get_artifact_version("email_tokenizer")
    tokenizer = tokenizer_artifact.load()
    
    model_info = client.get_artifact_version("model_info").load()
    max_sub = model_info.get("max_sub_len")
    max_body = model_info.get("max_body_len")
    subject = clean(subject)
    body = clean(body)

    subject_seq = pad_sequences(tokenizer.texts_to_sequences([subject]), maxlen=10)
    body_seq = pad_sequences(tokenizer.texts_to_sequences([body]), maxlen=max_body)
    
    return subject_seq, body_seq    