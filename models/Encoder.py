import torch
from torch import nn
from models.EncoderLayer import EncoderLayer
class Encoder(nn.Module):

    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner,
                 MHA_attn_dropout=0.1, MHA_fc_dropout=0.1, MHA_layerNorm=False,
                 FFN_dropout=0.1, FFN_layerNorm=True,
                 weight_init=True, CLS_learned=False, position_dropout=0.1, position_encoding=False, max_position_tokens=100,
                 relative_pe=False, window_size=4, window_depth=3, conv_patch=False, input_layerNorm=False, relative_pe_2D=False,
                 FFN_need=True):

        super().__init__()

        self.CLS_learned = CLS_learned
        if CLS_learned == True:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.position_encoding = position_encoding
        if position_encoding == True:
            self.position_dropout = nn.Dropout(position_dropout)
            self.position_enc = nn.Parameter(torch.randn(1, max_position_tokens, d_model))

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v,
                         MHA_attn_dropout=MHA_attn_dropout, MHA_fc_dropout=MHA_fc_dropout, MHA_layerNorm=MHA_layerNorm,
                         FFN_dropout=FFN_dropout, FFN_layerNorm=FFN_layerNorm,
                         relative_pe=relative_pe, window_size=window_size,
                         window_depth=window_depth, conv_patch=conv_patch,
                         relative_pe_2D=relative_pe_2D, FFN_need=FFN_need)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.input_layerNorm = input_layerNorm

        if weight_init == True:
            self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, enc_output, src_mask=None, return_attn=False, return_attn_v=False):

        enc_slf_attn_list = []
        enc_slf_v_list = []

        if self.input_layerNorm == True:
            enc_output = self.layer_norm(enc_output)

        if self.CLS_learned == True:
            CLS = self.cls_token.expand(enc_output.shape[0], -1, -1)
        else:
            CLS = torch.mean(enc_output, dim=1, keepdim=True)
        enc_output = torch.cat([CLS, enc_output], dim=1)

        if self.position_encoding == True:
            enc_output += self.position_enc[:, :enc_output.shape[1], :]
            enc_output = self.position_dropout(enc_output)

        for enc_layer in self.layer_stack:
            if return_attn_v == True:
                enc_output, enc_slf_attn, enc_slf_v = enc_layer(enc_output, slf_attn_mask=src_mask, return_attn=return_attn, return_attn_v=return_attn_v)
            else:
                enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask, return_attn=return_attn, return_attn_v=return_attn_v)
            enc_slf_attn_list += [enc_slf_attn] if return_attn or return_attn_v else []
            enc_slf_v_list += [enc_slf_v] if return_attn_v else []

        if return_attn_v == True:
            return enc_output, enc_slf_attn_list, enc_slf_v_list

        if return_attn:
            return enc_output, enc_slf_attn_list
        return enc_output