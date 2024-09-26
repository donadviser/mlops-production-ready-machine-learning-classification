import sys
from usvisa.logger import logging
from usvisa.exception import CustomException

from usvisa.components.data_ingestion import DataIngestion
from usvisa.components.data_validation import DataValidation
from usvisa.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
)
from usvisa.entity.artefact_entity import (
    DataIngestionArtefact,
    DataValidationArtefact,
)


class TrainPipeline:
    """
    This class is responsible for training the model
    """
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        
           
    def start_data_ingestion(self) -> DataIngestionArtefact:
        """
        Initiate data ingestion by exporting dataframe from mongodb feature store to csv file,
        then splitting the dataframe into train and test sets
        
        Output: DataIngestionArtefact
        On Failure:  Write an exception log and then raise an exception
        """
        try:
            logging.info("Starting data ingestion process")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def start_data_validation(self, data_ingestion_artefact: DataIngestionArtefact) -> DataValidationArtefact:
        """
        Validate the data to ensure it meets the specified criteria
        
        Output: DataValidationArtefact
        On Failure:  Write an exception log and then raise an exception
        """
        logging.info("Entered the start_data_validation method of TrainPipeline class")

        try:
            data_validation = DataValidation(data_ingestion_artefact=data_ingestion_artefact,
                                             data_validation_config=self.data_validation_config
                                             )

            data_validation_artefact = data_validation.initiate_data_validation()

            logging.info("Performed the data validation operation")

            logging.info(
                "Exited the start_data_validation method of TrainPipeline class"
            )

            return data_validation_artefact

        except Exception as e:
            raise CustomException(e, sys) 
        
        
    def run_pipeline(self) -> None:
        """
        Run the data ingestion, preprocessing, and training pipeline
        """
        try:
            logging.info("Starting training pipeline")
            
            data_ingestion_artefact = self.start_data_ingestion()
            
            data_validation_artefact = self.start_data_validation(data_ingestion_artefact)
            
            # TODO: Add your machine learning model training code here
            
            logging.info("Training pipeline completed successfully")
        except Exception as e:
            raise CustomException(e, sys)