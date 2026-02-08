import logging
from zenml import step
from src.model_building import BuildModel, TrainModel
import tensorflow as tf
from typing import Tuple, Annotated
import mlflow


@step(experiment_tracker="mlflow_tracker", enable_cache=False)
def building_model(tokenizer, data) -> Tuple[
    Annotated[tf.keras.Model,"model"],
    Annotated[dict, "model_info"],
    Annotated[float, "loss"]
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