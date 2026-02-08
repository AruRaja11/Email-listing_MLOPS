import logging
from zenml import step
import mlflow

@step
def load_production_model():
    model_uri = "models:/model/Production"
    model = mlflow.pyfunc.load_model(model_uri)
    return model