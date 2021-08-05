import os
import sys
import logging


logging_file = "logging.log"
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    handlers=[
        logging.FileHandler(logging_file),
        logging.StreamHandler()
    ]
)
getLogger = logging.getLogger

root_logger = getLogger("initial")
with open(logging_file, 'a') as f:
    f.write("\n" + "="*80 + "\n")
root_logger.info("Root Command: " + " ".join(sys.argv))
