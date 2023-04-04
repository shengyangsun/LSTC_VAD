import torch
from torch import nn
from utils.utils import weights_normal_init

class Classifier(nn.Module):
    def __init__(self, input_feature_dim, dropout_rate=0.6, weight_init=True):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(input_feature_dim,512), nn.ReLU(),nn.Dropout(dropout_rate),
                                     nn.Linear(512,32),nn.Dropout(dropout_rate),
                                     nn.Linear(32,2), nn.Softmax(dim=-1))

        if weight_init == True:
            self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x=x.view([-1,x.shape[-1]])
        logits=self.classifier(x)
        return logits