from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
from schedulers import *
from plotter import *
import copy


# 2 methods

# 1. Use shap on VAE hidden units to select top CpGs contributing to those units
# 2. Autoencoder feature selection 
