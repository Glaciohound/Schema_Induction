import os
import sys

from components.logging import getLogger

logger = getLogger("command")
command = " ".join(sys.argv[1:])
os.system(command)
