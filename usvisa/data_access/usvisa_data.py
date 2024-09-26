from usvisa.configuration.mongo_db_connection import MongoDBClient
from usvisa.constants import DATABASE_NAME
from usvisa.exception import CustomException
import pandas as pd
import sys
from typing import Optional
import numpy as np



class USvisaData:
    """
    This class help to export entire mongo db record as pandas dataframe
    """

    def __init__(self):
        """
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise CustomException(e, sys)
        

    def export_collection_as_dataframe(self,collection_name:str,database_name:Optional[str]=None)->pd.DataFrame:
        try:
            """
            export entire collectin as dataframe:
            return pd.DataFrame of collection
            """
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]

            df = (pd.DataFrame(list(collection.find()))
                  .drop(columns=["_id"], errors="ignore")  # Safely drop "_id" if it exists
                  .replace("na", np.nan) # Replace "na" with NaN
                )  
            return df
        except Exception as e:
            raise CustomException(e, sys)