import os
import sys
import logging


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    handlers=[
        logging.FileHandler("logging.log"),
        logging.StreamHandler()
    ]
)
getLogger = logging.getLogger

root_logger = getLogger("initial")
root_logger.info("")
root_logger.info("=" * 60)
root_logger.info("Python Command: " + " ".join(sys.argv))
