import os
import sys
import logging
from datetime import datetime

logging_str = "[%(asctime)s: %(name)s: %(levelname)s: %(funcName)s: %(lineno)d: %(message)s]"

log_dir = "logs"
#To gemerate a new log file each day
log_file_name =f"usvisa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(log_dir,log_file_name)
os.makedirs(log_dir, exist_ok=True)


logging.basicConfig(
    level= logging.INFO,
    format= logging_str,

    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logging = logging.getLogger("usvisaLogger")
