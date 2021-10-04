import os
import sys
import logging
from inspect import currentframe, getouterframes


logging_file = "logging.log"
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    handlers=[
        logging.FileHandler(logging_file),
        logging.StreamHandler()
    ]
)


class Logger:
    def __init__(self):
        pass

    def info(self, text, level_offset=0):
        frameinfo = getouterframes(currentframe())[1+level_offset]
        filename = frameinfo.filename.split('/')[-1].split('.')[0]
        lineno = frameinfo.lineno
        _inner_logger = logging.getLogger(f"{filename}:{lineno}")
        _inner_logger.info(text)


logger = Logger()


with open(logging_file, 'a') as f:
    f.write("\n" + "="*80 + "\n")
logger.info("Root Command: " + " ".join(sys.argv))
