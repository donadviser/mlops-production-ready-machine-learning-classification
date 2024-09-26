import os
from os import environ
import datetime

TIMESTAMP: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# MongoDB related Constants
DATABASE_NAME = 'US_VISA'
COLLECTION_NAME = 'visa_data'
MONGODB_URL_KEY = environ["MOGODB_USVISA_URL"]


# S3 related Constants
PIPELINE_NAME: str = "usvisa"
ARTEFACTS_DIR: str = "artefacts"


# Model related constants
FILE_NAME: str = "usvisa.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
TARGET_COLUMN = "case_status"
CURRENT_YEAR =  datetime.date.today().year
MODEL_FILE_NAME: str = "model.pkl"
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"



MODEL_CONFIG_FILE = os.path.join("config", "model.yaml") 
# CONFIG_FILE_PATH = os.path.join("config", "config.yaml") 
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = COLLECTION_NAME
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2