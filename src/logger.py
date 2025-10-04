#logging configuration what all things to be logged
import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)#even though there is a folder keep on adding file into it

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    #logging level can be changed to DEBUG,WARNING,ERROR,CRITICAL
)
# if __name__=="__main__":
#     logging.info("Logging has started")