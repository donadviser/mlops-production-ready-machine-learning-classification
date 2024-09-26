from dataclasses import dataclass


@dataclass
class DataIngestionArtefact:
    trained_file_path:str 
    test_file_path:str 
