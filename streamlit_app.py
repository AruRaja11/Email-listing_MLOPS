import streamlit as st 

import json 

from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import run_deployment
from pipelines.utils import get_data_for_test

import pandas as pd 
import numpy as np

from zenml.client import Client


def main():
    subject = st.text_input(label="Subject")
    body = st.text_input(label="Body")

    if st.button("predict"):
        service = prediction_service_loader(
            pipeline_name='continuous_deployment_pipeline',
            pipeline_step_name = "mlflow_model_deployer_step"
        )
        if service is None:
            st.write(
                "no service found"
            )
            main()

        subject_seq, body_seq = get_data_for_test(subject, body)

        data = pd.DataFrame({
            "subject": [subject_seq.tolist()[0]],
            "body": [body_seq.tolist()[0]]
        })
        model = Client().get_artifact_version("model").load()
        predictions = model.predict([subject_seq, body_seq])
        
        output = np.argmax(predictions)

        st.success(
           f"the predicted output is {output}"
        )


if __name__ == "__main__":
    main()