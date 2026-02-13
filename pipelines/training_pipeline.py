import logging 
from zenml import pipeline
from steps.ingestion import ingest_data
from steps.preprocessing import preprocess_data
from steps.building import building_model


@pipeline
def training_line(data_path:str):
    try:
        data = ingest_data(data_path)
        tokenizer, encoder, data = preprocess_data(data)
        model,model_info, history_loss = building_model(tokenizer, data)

        return model

    except Exception as e:
        logging.error(f"Error in pipeline: {e}")
        raise e
