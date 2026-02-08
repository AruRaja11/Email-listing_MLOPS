import numpy as np
import pandas as pd
from typing import Annotated, Tuple
from pydantic import BaseModel
import string
import json 

from zenml.client import Client
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from pipelines.utils import get_data_for_test

# Existing steps from your project
from steps.ingestion import ingest_data
from steps.preprocessing import preprocess_data
from steps.building import building_model

# Docker configurations
docker_settings = DockerSettings(required_integrations=[MLFLOW, TENSORFLOW])

# --- CONFIGURATION CLASSES ---

class DeploymentTriggerConfig(BaseModel):
    min_loss: float = 0.9

# --- CUSTOM STEPS ---

@step(enable_cache=False)
def dynamic_importer(subject:str, body:str) -> Tuple[Annotated[np.ndarray, "subject"], Annotated[np.ndarray, "body"]]:
    subject_seq, body_seq = get_data_for_test(subject, body)
    return subject_seq, body_seq

@step
def deployment_trigger(
    loss: float,
    config: DeploymentTriggerConfig
) -> bool:
    """Decides if the model is good enough to deploy based on loss."""
    return loss <= config.min_loss

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model"
) -> MLFlowDeploymentService:
    """Finds the active MLflow deployment server."""
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()

    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow deployment service found for pipeline {pipeline_name}. "
            f"Ensure the continuous_deployment_pipeline has run successfully."
        )
    return existing_services[0]



@step
def predictor(
    service: MLFlowDeploymentService,
    subject: np.ndarray,
    body: np.ndarray
) -> int: 
    service.start(timeout=10)

    model = Client().get_artifact_version("model").load()
    prediction = model.predict([subject, body])
    print(prediction)
    return int(np.argmax(prediction[0]))

# --- PIPELINES ---

@pipeline(enable_cache=True, settings={'docker': docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_loss: float = 0.9,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    """The Automated Assembly Line: Train -> Evaluate -> Deploy."""
    
    # 1. Ingest
    data = ingest_data(data_path)
    
    # 2. Preprocess (returns Tokenizer, Encoder, and DataFrame)
    tokenizer, encoder, preprocessed_df = preprocess_data(data)
    
    # 3. Build & Train
    model, model_info, loss = building_model(tokenizer, preprocessed_df)
    
    # 5. Safety Gate
    deployment_decision = deployment_trigger(loss, config=DeploymentTriggerConfig(min_loss=min_loss))

    # 6. Deploy to MLflow
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout
    )

@pipeline(enable_cache=False, settings={'docker': docker_settings})
def inference_pipeline(subject:str, body:str, pipeline_name: str, pipeline_step_name: str):
    """The Prediction Service: Locate the model and use it."""
    subject_seq, body_seq = dynamic_importer(subject=subject, body=body)

    service = prediction_service_loader(
        pipeline_name=pipeline_name,    
        pipeline_step_name=pipeline_step_name,
        running=True
    )

    
    # 2. Run prediction (Placeholder for actual Gmail text)
    prediction = predictor(service=service, subject=subject_seq, body=body_seq)
    return prediction