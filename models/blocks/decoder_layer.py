"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        
        # 1. Masked Self-Attention (디코더 스스로를 쳐다봄)
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, 
                                                    dropout=drop_prob, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        # 2. Encoder-Decoder Attention (인코더의 정보를 가져옴)
        self.enc_dec_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, 
                                                        dropout=drop_prob, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        # 3. Positionwise Feed Forward (내장 Linear 모듈 조합)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(ffn_hidden, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        # --- 1. Masked Self Attention ---
        # trg_mask는 미래 토큰을 가리는용도 (Look-ahead mask)
        residual = dec
        # nn.MultiheadAttention은 (output, attn_weights)를 반환함
        x, _ = self.self_attention(query=dec, key=dec, value=dec, attn_mask=trg_mask)
        x = self.norm1(x + self.dropout1(x))

        # --- 2. Encoder-Decoder Attention ---
        if enc is not None:
            residual = x
            # Query는 디코더(x), Key와 Value는 인코더(enc)에서 가져옴
            # src_mask는 인코더의 패딩 토큰을 무시하는 용도
            x, _ = self.enc_dec_attention(query=x, key=enc, value=enc, attn_mask=src_mask)
            x = self.norm2(x + self.dropout2(x))

        # --- 3. Positionwise Feed Forward ---
        residual = x
        x = self.ffn(x)
        x = self.norm3(x + self.dropout3(x))
        
        return x
