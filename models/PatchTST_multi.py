import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from einops.layers.torch import Rearrange
from einops import rearrange, reduce
import torch.nn.functional as F
import sys





class Model(nn.Module):
    def __init__(self, configs, stride=8):
        super(Model, self).__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_sizes1 = configs.patch_sizes1
        self.patch_sizes2 = configs.patch_sizes2
        self.patch_sizes3 = configs.patch_sizes3
        self.enc_in = configs.enc_in
        self.dec_in = configs.dec_in
        
        padding = stride
     
        
        # patching and embedding
        self.patch_embedding1 = PatchEmbedding(
            configs.d_model, self.patch_sizes1, stride, padding, configs.dropout)
        
        self.patch_embedding2 = PatchEmbedding(
            configs.d_model, self.patch_sizes2, stride, padding, configs.dropout)
        
        self.patch_embedding3 = PatchEmbedding(
            configs.d_model, self.patch_sizes3, stride, padding, configs.dropout)
        
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        
        self.out = nn.Linear(11776, self.pred_len) 
        self.dropout = nn.Dropout(0.05)    
        
        
    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B,_,C = x.shape
        
        # Normalization from Non-stationary Transformer
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        
        x = rearrange(x, "b l c -> b c l") # x:[32, 96, 8]-->[32, 8, 96]
     
        enc_out1, n_vars = self.patch_embedding1(x) 
        enc_out2, n_vars = self.patch_embedding2(x) 
        enc_out3, n_vars = self.patch_embedding3(x)
        
        output1,_ = self.encoder(enc_out1) # [256, 10, 512]
        
        output2,_ = self.encoder(enc_out2)  # Skip connection [256, 13, 512]
        
        output3,_ = self.encoder(enc_out3)  # Skip connection [256, 13, 512]
        

        
        out1 = output1.reshape(B*C,-1) # [256, 5120]
        out2 = output2.reshape(B*C,-1) # [256, 6656]
        out3 = output3.reshape(B*C,-1)
        
        concat_output = torch.cat([out1, out2, out3], dim=1) # [256, 11776]
        output = rearrange(concat_output, "(b c) l -> b c l", c = self.enc_in) # [32, 8, 11776]
        out_put = self.out(output) # # [32, 8, pre_len]
        out_put = self.dropout(out_put)
        
        dec_out = out_put.permute(0,2,1)

        
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        
        return dec_out