import logging
from zenml import step
from src.model_building import BuildModel, TrainModel
import tensorflow as tf
from typing import Tuple, Annotated
from tensorflow.keras.models import Model
import mlflow


from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

@step(experiment_tracker="mlflow_tracker", enable_cache=False)
def building_model(tokenizer: Tokenizer, data: pd.DataFrame) -> Tuple[
    Model,
    dict,
    float
]:
    try:
        builder = BuildModel()
        trainer = TrainModel()

        vocab_size = len(tokenizer.word_index) + 1
        max_sub_len = len(data['subject_seq'].iloc[0])
        max_body_len = len(data['body_seq'].iloc[0])
        model_info = {
            "max_subject_len": max_sub_len,
            "max_body_len": max_body_len,
            "vocab_size": vocab_size,
        }

        model = builder.build(vocab_size, max_sub_len, max_body_len)
        mlflow.tensorflow.log_model(model=model, artifact_path="model")
        
        # model training
        loss = trainer.train(model, data,epoch=5)
        return model, model_info, loss
    
    except Exception as e:
        logging.error(f"Error while building DL model: {e}")
        raise e