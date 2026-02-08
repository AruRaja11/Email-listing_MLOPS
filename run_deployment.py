from pipelines.deployment_pipeline import continuous_deployment_pipeline, inference_pipeline
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (MLFlowModelDeployer, )
import click
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from typing import cast
from zenml.integrations.mlflow.services import MLFlowDeploymentService

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

@click.command()
@click.option(
    "--config",
    "-c",
    type = click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default = DEPLOY_AND_PREDICT,
    help= "optionally you can choose only to run deployment"
    "pipeline to train and deploy (`deploy')"
    "pipeline to predict (`predict`)"
    "default is (`predict_and_deploy`)"

)

@click.option(
    "--min_loss",
    default=0.9,
    help = "minimum accuracy to deploy the model"
)

def run_deployment(config:str, min_loss:float):
    mlflow_deployment_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT

    if deploy:
        data_path = "/home/arun-raja/Documents/Datasets/email_listing.csv"
        continuous_deployment_pipeline(data_path, min_loss=min_loss, workers=3, timeout=60)
    if predict:
        inference_pipeline(
            subject = "Google is hiring, Arun - Youre a top match!",
            body="Hi Arun, Google is hiring for the role of Product Support Engineer! Congrats, your profile matches this opportunity!",
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name = "mlflow_model_deployer_step"
        )   

    print(
        "you can run:\n"
        f"[italic green] mlflow ui --backend-store-ui"
    )

    existing_services = mlflow_deployment_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model"
    )

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(
                f"running mlflow service"
            )

        elif service.is_failed:
            print(f"mlfow server failed")
        
    else:
        print("no mlflow production found")


if __name__ == "__main__":
    run_deployment()