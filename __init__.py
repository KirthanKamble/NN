# Add the parent directory of 'NN' to the Python path
import sys
import os
sys.path.append(os.path.abspath('..'))

from NN.Layers import *
from NN.LossFunctions import *
from NN.Activations import *
from NN.Networks import *
