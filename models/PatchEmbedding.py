import torch
from torch import nn

class PatchEmbedding(nn.Module):

    def __init__(self, embed_dim=2048, CLS_learned=False):
        super().__init__()
        self.CLS_learned = CLS_learned
        if CLS_learned == True:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, feats):

        if self.CLS_learned == False:
            CLS = torch.mean(feats, dim=1, keepdim=True)
        else:
            CLS = self.cls_token.expand(feats.shape[0], -1, -1)
        feats = torch.cat([CLS, feats], dim=1)
        return feats