import sys
import pandas as pd
import numpy as np
import matplotlib
import scipy as sp
import IPython
from IPython import display

import random
import time 

import warnings
warnings.filterwarnings('ignore')

from subprocess import check_output
print(check_output(["ls", "titanic"]).decode("utf8"))
