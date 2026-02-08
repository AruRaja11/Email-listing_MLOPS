import logging
import string
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Annotated
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class DataPreProcessorStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class CleanData(DataPreProcessorStrategy):
    """
    Strategy for cleaning email text.
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        def cleaner(text):
            if not isinstance(text, str):
                return ""
            text = text.lower()
            # Remove punctuation
            text = text.translate(str.maketrans("", "", string.punctuation))
            # Basic tokenization and filtering
            words = text.split()
            words = [word for word in words if len(word) > 1 and word.isalpha()]
            return " ".join(words)
        
        data['subject'] = data['subject'].apply(cleaner)
        data['body'] = data['body'].apply(cleaner)
        return data

class TokenizeData:
    """
    Handles tokenization and encoding.
    Returns the tools (tokenizer, encoder) and the processed dataframe.
    """
    def handle_tokenization(self, data: pd.DataFrame) -> Tuple[
        Annotated[Tokenizer,'tokenizer'], 
        Annotated[LabelEncoder,'encoder'], 
        Annotated[pd.DataFrame, "dataframe"]
        ]:
        tokenizer = Tokenizer()
        encoder = LabelEncoder()
        
        # Combine subject and body to build the full vocabulary
        combined_text = data['subject'] + " " + data['body']
        tokenizer.fit_on_texts(combined_text)

        # Convert text to sequences
        subject_seq = tokenizer.texts_to_sequences(data['subject'])
        body_seq = tokenizer.texts_to_sequences(data['body'])

        # Calculate max lengths (needed for model building)
        max_sub_len = max(len(w) for w in subject_seq)
        max_body_len =max(len(w) for w in body_seq)
        
        # Pad sequences
        data['subject_seq'] = pad_sequences(subject_seq, maxlen=max_sub_len).tolist()
        data['body_seq'] = pad_sequences(body_seq, maxlen=max_body_len).tolist()

        # Encode target labels
        data['target'] = encoder.fit_transform(data['category']).tolist()

        return tokenizer, encoder, data
