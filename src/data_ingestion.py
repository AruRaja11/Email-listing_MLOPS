import logging
import pandas as pd
from datasets import load_dataset

class DataIngestion():
    def get_data(self, data_path:str)->pd.DataFrame:
        """
        read the data from the dataset path
        
        :param data_path: Path of the dataframe
        :type data: string
        :return: read dataframe
        :rtype: DataFrame
        """
        ds = pd.read_csv(data_path)
        return ds
    

