import logging 
from zenml import step
import pandas as pd
from src.data_ingestion import DataIngestion

@step
def ingest_data(data_path:str) -> pd.DataFrame: 
    try:
        """
        reading data from datapath
        
        :param data_path: location of the data 
        :type data_path: str
        :return: dataframe read by pandas
        :rtype: DataFrame
        """

        ingestor = DataIngestion()
        data = ingestor.get_data(data_path)

        return data
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e