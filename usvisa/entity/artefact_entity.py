from dataclasses import dataclass


@dataclass
class DataIngestionArtefact:
    trained_file_path:str 
    test_file_path:str 


@dataclass
class DataValidationArtefact:
    validation_status:bool
    message: str
    drift_report_file_path: str
    
@dataclass
class DataTransformationArtefact:
    transformed_object_file_path:str 
    transformed_train_file_path:str
    transformed_test_file_path:str

@dataclass
class ClassificationMetricArtefact:
    f1_score:float
    precision_score:float
    recall_score:float   
    
@dataclass
class ModelTrainerArtefact:
    trained_model_file_path:str 
    metric_artefact:ClassificationMetricArtefact