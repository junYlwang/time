import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d
from torch.nn.utils import weight_norm

from .utils import init_weights, get_padding
from .stochastic_fsq import FSQ, IFSQ
from .backbones import ConvNeXtBlock, DownSamplingBlock


import torch
import torch.nn as nn
from torch import Tensor

class InvertibleLayerNorm(nn.Module):
    """可逆的LayerNorm模块，专门处理 (B, D, T) 格式"""
    
    def __init__(self, num_dims, eps=1e-5):
        super().__init__()
        self.num_dims = num_dims
        self.eps = eps
        
        # 可学习参数，针对特征维度 D
        self.weight = nn.Parameter(torch.ones(num_dims))
        self.bias = nn.Parameter(torch.zeros(num_dims))
        
        # 存储统计信息
        self.register_buffer('current_mean', None, persistent=False)
        self.register_buffer('current_std', None, persistent=False)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        输入: x (B, D, T)
        输出: normalized (B, D, T)
        """
        B, D, T = x.shape
        
        # LayerNorm 标准操作：在 D 维度上计算均值和标准差
        # 注意：dim=1 对应 D 维度
        self.current_mean = x.mean(dim=1, keepdim=True)  # (B, 1, T)
        variance = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, T)
        self.current_std = torch.sqrt(variance + self.eps)
        
        # 执行标准化
        normalized = (x - self.current_mean) / self.current_std
        
        # 应用可学习参数 (D 维度对齐)
        # weight 和 bias 形状为 (num_dims,) -> 变为 (1, D, 1) 进行广播
        weight = self.weight.view(1, D, 1)
        bias = self.bias.view(1, D, 1)
        
        return weight * normalized + bias
    
    def inverse(self, normalized_x: Tensor) -> Tensor:
        """
        输入: normalized_x (B, D, T)
        输出: original (B, D, T)
        """
        if self.current_mean is None or self.current_std is None:
            raise RuntimeError("必须先调用forward方法")
            
        B, D, T = normalized_x.shape
        weight = self.weight.view(1, D, 1)
        bias = self.bias.view(1, D, 1)
        
        # 逆变换
        denormalized = (normalized_x - bias) / weight
        return denormalized * self.current_std + self.current_mean


class Encoder(torch.nn.Module):
    def __init__(self, h):
        super(Encoder, self).__init__()

        self.input_channels=h.num_mels
        self.h=h
        self.in_dim = getattr(h, "encoder_in_dim", 128)
        self.num_layers = getattr(h, "encoder_num_layers", [4, 8, 12])
        self.out_dim = self.in_dim * (2**len(h.down_ratio)) #1024
        self.adanorm_num_embeddings=None
        self.layer_scale_init_value =  [1/num_layer for num_layer in self.num_layers]

        embed_ks = getattr(h, "encoder_embed_kernel_size", 7)
        self.embed_mel = nn.Conv1d(self.input_channels, self.in_dim, kernel_size=embed_ks, padding=embed_ks//2)
        self.norm_mel = nn.LayerNorm(self.in_dim, eps=1e-6)
        self.convnext_mel = nn.ModuleList()
        for i in range(len(self.num_layers)):
            for _ in range(self.num_layers[i]):
                cur_dim = self.in_dim * (2**i)
                cur_intermediate_dim  = cur_dim*4
                self.convnext_mel.append(ConvNeXtBlock(cur_dim, cur_intermediate_dim, self.layer_scale_init_value[i], self.adanorm_num_embeddings))
            self.convnext_mel.append(DownSamplingBlock(cur_dim, cur_dim*2, self.h.mel_Encoder_convnext_kernel_size[i], self.h.down_ratio[i],
                                                  padding=(self.h.mel_Encoder_convnext_kernel_size[i] - self.h.down_ratio[i]) // 2))

        self.final_layer_norm_mel = nn.LayerNorm(self.out_dim, eps=1e-6)
        self.apply(self._init_weights)

        self.out_mel = torch.nn.Linear(self.out_dim, h.mel_Encoder_channel)

        self.mel_Encoder_output_conv = weight_norm(Conv1d(h.mel_Encoder_channel, h.mel_Encoder_channel//4, h.mel_Encoder_output_kernel_size, 1, 
                                                  padding=get_padding(h.mel_Encoder_output_kernel_size, 1)))
        
        self.latent_output_conv = weight_norm(Conv1d(h.mel_Encoder_channel//4, h.latent_dim, h.latent_output_conv_kernel_size, 1, 
                                                  padding=get_padding(h.latent_output_conv_kernel_size, 1)))

        
        self.mel_Encoder_output_conv.apply(init_weights)
        self.latent_output_conv.apply(init_weights)

        self.quantizer_1 = IFSQ(
            levels=h.levels_1,
            dim=h.latent_dim,
            channel_first = True,
            stochastic=h.stochastic
        )

        self.quantizer_2 = IFSQ(
            levels=h.levels_2,
            dim=h.latent_dim,
            channel_first = True,
            stochastic=h.stochastic
        )

        self.layernorm_1 = InvertibleLayerNorm(h.latent_dim)
        self.layernorm_2 = InvertibleLayerNorm(h.latent_dim)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, mel):

        mel_encode = self.embed_mel(mel)
        mel_encode = self.norm_mel(mel_encode.transpose(1, 2))
        mel_encode = mel_encode.transpose(1, 2)
        for conv_block in self.convnext_mel:
            mel_encode = conv_block(mel_encode)
        mel_encode = self.final_layer_norm_mel(mel_encode.transpose(1, 2))
        mel_encode = self.out_mel(mel_encode).transpose(1, 2)
        mel_encode = F.gelu(mel_encode)
        mel_encode = self.mel_Encoder_output_conv(mel_encode)
        mel_encode = F.gelu(mel_encode)
        latent = self.latent_output_conv(mel_encode)
        
        latent_1, codes_1 = self.quantizer_1(self.layernorm_1(latent))
        latent_1 = self.layernorm_1.inverse(latent_1)

        latent_2, codes_2 = self.quantizer_2(self.layernorm_2(latent - latent_1.detach()))
        latent_2 = self.layernorm_2.inverse(latent_2)

        quantized_latent = latent_1 + latent_2
        codes = torch.stack([codes_1, codes_2], dim=1)
        
        return quantized_latent, codes