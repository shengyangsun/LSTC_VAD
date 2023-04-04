from torch import nn
from models.MultiHeadAttention import MultiHeadAttention
from models.FFN import PositionwiseFeedForward
class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, MHA_attn_dropout=0.1, MHA_fc_dropout=0.1,
                 MHA_layerNorm=False, FFN_dropout=0.1, FFN_layerNorm=True, return_attn=False,
                 relative_pe=False, window_size=4, window_depth=3, conv_patch=False,
                 relative_pe_2D=False, FFN_need=True):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v,
                                           attn_dropout=MHA_attn_dropout, fc_dropout=MHA_fc_dropout,
                                           layerNorm=MHA_layerNorm, relative_pe=relative_pe, window_size=window_size,
                                           window_depth=window_depth, conv_patch=conv_patch, relative_pe_2D=relative_pe_2D)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=FFN_dropout, layerNorm=FFN_layerNorm)
        self.FFN_need = FFN_need

    def forward(self, enc_input, slf_attn_mask=None, return_attn=False, return_attn_v=False):
        if return_attn_v == True:
            enc_output, enc_slf_attn, enc_slf_v = self.slf_attn(
                enc_input, enc_input, enc_input, mask=slf_attn_mask, return_attn=return_attn, return_attn_v=return_attn_v)
        else:
            enc_output, enc_slf_attn = self.slf_attn(
                enc_input, enc_input, enc_input, mask=slf_attn_mask, return_attn=return_attn, return_attn_v=return_attn_v)
        if self.FFN_need == True:
            enc_output = self.pos_ffn(enc_output)
        if return_attn_v == True:
            return enc_output, enc_slf_attn, enc_slf_v
        else:
            return enc_output, enc_slf_attn