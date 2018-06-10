import os
import sys


RUN_IN_SMALL_MODE = 'RUN_IN_SMALL_MODE' in os.environ and os.environ['RUN_IN_SMALL_MODE'] != '0'
if RUN_IN_SMALL_MODE:
    print('RUN_IN_SMALL_MODE', file=sys.stderr)
