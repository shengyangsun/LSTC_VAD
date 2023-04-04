import sys
sys.path.extend(['..'])
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils.utils import weights_normal_init

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, relative_pe=False, window_size=4):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, layerNorm=False,
                 attn_dropout=0.1, fc_dropout=0.1, relative_pe=False, window_size=3,
                 window_depth=3, conv_patch=False, relative_pe_2D=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.layerNorm_flag = layerNorm
        self.d_model = d_model


        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.dropout = nn.Dropout(fc_dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.temperature = d_k ** 0.5
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.relative_pe_2D = relative_pe_2D
        self.relative_pe = relative_pe
        self.window_size = window_size
        self.window_depth = window_depth
        if relative_pe == True:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_depth - 1) * (2 * window_size - 1) * (2 * window_size - 1), n_head))
            # get pair-wise relative position index for each token inside the window
            coord_d = torch.arange(self.window_depth)
            coords_h = torch.arange(self.window_size)
            coords_w = torch.arange(self.window_size)
            coords = torch.stack(torch.meshgrid([coord_d, coords_h, coords_w]))  # 3, Wd, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww,Wd*Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_depth - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size - 1
            relative_coords[:, :, 2] += self.window_size - 1

            relative_coords[:, :, 0] *= (2 * self.window_size - 1) * (2 * self.window_size - 1)
            relative_coords[:, :, 1] *= (2 * self.window_size - 1)
            relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        if relative_pe_2D == True:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), n_head))
            coords_h = torch.arange(self.window_size)
            coords_w = torch.arange(self.window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size - 1
            relative_coords[:, :, 1] += self.window_size - 1
            relative_coords[:, :, 0] *= 2 * self.window_size - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
            nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)


    def forward(self, q, k, v, mask=None, return_attn=False, return_attn_v=False):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        if self.relative_pe == True:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:len_q-1, :len_q-1].reshape(-1)].reshape(
                len_q-1, len_q-1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn[:, :, 1:, 1:] += relative_position_bias.unsqueeze(0)

        if self.relative_pe_2D == True:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn[:, :, 1:, 1:] += relative_position_bias.unsqueeze(0)

        attn = self.attn_dropout(F.softmax(attn, dim=-1))
        attn_output = torch.matmul(attn, v)
        q = attn_output
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        if self.layerNorm_flag == True:
            q = self.layer_norm(q)
        if return_attn_v == True:
            return q, attn, v
        if return_attn == False:
            return q, None
        else:
            return q, attn
