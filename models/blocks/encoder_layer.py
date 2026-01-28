"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


# We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized
# Sub-layer란 Multi-Head Attention과 Feed Forward를 의미
# FFN 연산이 끝난 직후, **잔차 연결(Residual Connection)**로 더해지기 바로 직전에 드롭아웃을 적용
# 과적합(Overfitting) 방지: FFN은 보통 $d_{model}$(예: 512)보다 훨씬 큰 $d_{ff}$(예: 2048) 차원으로 확장했다가 
# 다시 줄이는 구조입니다. 이 과정에서 파라미터가 몰려있어 과적합이 일어나기 쉽기 때문에 드롭아웃으로 규제를 주는 것

'''
class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)  
    def forward(self, x, src_mask):
            # 1. compute self attention
            _x = x
            x = self.attention(q=x, k=x, v=x, mask=src_mask)
'''
import torch
from torch import nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        
        # 1. 파이토치 내장 MultiheadAttention
        # batch_first=True로 설정하면 (batch, seq, feature) 순서를 유지할 수 있습니다.
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, 
                                               dropout=drop_prob, batch_first=True)
        
        # 2. 파이토치 내장 LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 3. 파이토치 모듈 조합으로 만든 FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(ffn_hidden, d_model)
        )
        
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. Self Attention
        # nn.MultiheadAttention은 (output, weights)를 반환하므로 [0]으로 결과만 가져옵니다.
        # mask는 attn_mask 파라미터에 전달합니다.
        residual = x
        x, _ = self.attention(query=x, key=x, value=x, attn_mask=src_mask)
        
        # 2. Add & Norm
        x = self.dropout1(x)
        x = self.norm1(x + residual)
        
        # 3. Feed Forward
        residual = x
        x = self.ffn(x)
        
        # 4. Add & Norm
        x = self.dropout2(x)
        x = self.norm2(x + residual)
        
        return x
