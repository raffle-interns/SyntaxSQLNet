import torch
import numpy as np
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    
from utils.lstm import PackedLSTM
from utils.attention import ConditionalAttention
from models.agg_predictor import AggPredictor

class LimitValuePredictor(AggPredictor):
    """
    This module is identical to AggPredictor, so we inherit.
    """
    def __init__(self, num=10, *args, **kwargs):
        super(LimitValuePredictor, self).__init__(*args, **kwargs, num=num)

    def predict(self, *args):
        return AggPredictor.predict(self, *args) + 1

    def loss(self, prediction, batch):
        truth = batch[self.name].to(prediction.device).long() - 1

        # Match dimensions of input and target to softmax expectations
        if len(prediction.shape) == 1: prediction = prediction.unsqueeze(0) # (N, C)
        truth = truth.squeeze(dim=-1) # (N)

        return self.cross_entropy(prediction, truth)

    def accuracy(self, prediction, batch):
        truth =  batch[self.name].to(prediction.device).squeeze(1).long() - 1
        batch_size = len(truth)

        # Predict number of columns as the argmax of the scores
        prediction = torch.argmax(prediction, dim=-1)
        
        # Compute accuracy
        accuracy = (prediction==truth).sum().float()/batch_size
        
        return accuracy.detach().cpu().numpy()