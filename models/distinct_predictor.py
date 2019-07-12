import torch
import numpy as np
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    
from utils.lstm import PackedLSTM
from utils.attention import ConditionalAttention
from models.agg_predictor import AggPredictor

class DistinctPredictor(AggPredictor):
    """
    This module is identical to AggPredictor, so we inherit.
    """
    def __init__(self, num=2, *args, **kwargs):
        super(DistinctPredictor, self).__init__(*args, **kwargs, num=num)
