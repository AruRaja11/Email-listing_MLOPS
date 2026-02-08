import logging
import numpy as np
import pandas as pd
from zenml import step
from typing import Tuple, Annotated # Added Annotated
from tensorflow.keras.preprocessing.text import Tokenizer
from src.data_preprocessing import CleanData, TokenizeData
from sklearn.preprocessing import LabelEncoder


@step
def preprocess_data(data: pd.DataFrame) -> Tuple[
        Annotated[Tokenizer, "email_tokenizer"],
        Annotated[LabelEncoder, "encoder"],
        Annotated[pd.DataFrame, "preprocessed_data"]
    ]:
    try:
        clean_dataframe = CleanData()
        cleaned_data = clean_dataframe.handle_data(data)

        tokenizing_logic = TokenizeData()
        tokenizer, encoder, processed_df = tokenizing_logic.handle_tokenization(cleaned_data)
        
        return tokenizer, encoder, processed_df

    except Exception as e:
        logging.error(f"Error while preprocessing and tokenizing data: {e}")
        raise e