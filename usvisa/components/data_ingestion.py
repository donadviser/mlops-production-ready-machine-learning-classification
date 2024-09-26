import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from usvisa.entity.config_entity import DataIngestionConfig
from usvisa.entity.artefact_entity import DataIngestionArtefact
from usvisa.exception import CustomException
from usvisa.logger import logging
from usvisa.data_access.usvisa_data import USvisaData

class DataIngestion:
    """
    This class is responsible for data ingestion and preprocessing
    """
    def __init__(self, data_ingestion_config: DataIngestionConfig=DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
            self.usvisa_data = USvisaData()
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def export_data_into_feature_store(self) -> pd.DataFrame:
        """
        Export the dataframe from mongodb feature store to csv file 
        """
        try:
            logging.info("Exporting data from MongoDB into feature store")
            
            
            collection_name = self.data_ingestion_config.collection_name
            
            df = self.usvisa_data.export_collection_as_dataframe(collection_name=collection_name)
            logging.info(f"Shae of dataframe: {df.shape}")
            
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            
            df.to_csv(feature_store_file_path, index=False, header=True)
            return df
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def split_data_as_train_test(self, dataframe:pd.DataFrame) -> None:
        """
        Split the dataframe into train and test sets
        
        Output: Folder is created in s3 bucket
        on Failure:  Write an exception log and then raise an exception
        """
        
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")
        try:
            train_set, test_set =  train_test_split(dataframe, 
                                                    test_size=self.data_ingestion_config.train_test_split_ratio, 
                                                    random_state=42
                                                    )
            logging.info("Completed train test split on the dataframe")
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            logging.info(f"Saving training data into file path: {self.data_ingestion_config.training_file_path}")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            
            logging.info(f"Shape of training data: {train_set.shape}")
            logging.info(f"Shape of testing data: {test_set.shape}")
            logging.info("train set and test set exported to file path")
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def initiate_data_ingestion(self) -> DataIngestionArtefact:
        """
        Method Name: initiate_data_ingestion
        Description: Initiate data ingestion by exporting dataframe from mongodb feature store to csv file,
        then splitting the dataframe into train and test sets
        
        Output: DataIngestionArtefact
        On Failure:  Write an exception log and then raise an exception
        """
        
        logging.info("Entering initiate_data_ingestion method of Data_Ingestion class")
        try:
            logging.info("Starting data ingestion process")
            df = self.export_data_into_feature_store()
            logging.info("Obtained the data from MongoDB")
            
            self.split_data_as_train_test(df)
            logging.info("Performed train test split on the dataset")
            
            logging.info("Data ingestion process completed successfully")
            data_ingestion_artefact =  DataIngestionArtefact(trained_file_path=self.data_ingestion_config.training_file_path, 
                                       test_file_path=self.data_ingestion_config.testing_file_path)
            
            logging.info(f"Data ingestion artefact: {data_ingestion_artefact}")
            return data_ingestion_artefact
        except Exception as e:
            raise CustomException(e, sys)