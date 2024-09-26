from usvisa.logger import logging
from usvisa.exception import CustomException
import sys
from os import environ
from usvisa.pipeline.training_pipeline import TrainPipeline


obj = TrainPipeline()
obj.run_pipeline()