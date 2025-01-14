from functools import partial
from typing import List

import einops
import torch
from einops.layers.torch import EinMix, Rearrange, Reduce
from torch import nn
from torch.nn import RMSNorm
from torch.nn import functional as F


def make_list(input, length, repeat=True, pad_mode='last'):
    match input:
        case None:
            return [None] * length
        case list() | tuple():
            if pad_mode == 'last':
                input = input + [input[-1]] * (length - len(input))
            elif pad_mode == 'ones':
                input = input + [1] * (length - len(input))
            elif pad_mode == 'zeros':
                input = input + [0] * (length - len(input))
            else:
                input = input + [None] * (length - len(input))
        case _:
            if repeat:
                input = [input] * length
            else:
                input = [input] + [None] * (length - 1)
                
    input = [None if x == 'null' else x for x in input]
    return input


class LayerNormFeature(nn.LayerNorm):    
    def forward(self, input):
        return super().forward(input.moveaxis(-1, -2)).moveaxis(-1, -2)
    
    
class RmsNormFeature(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.rms_norm = RMSNorm(features)
    
    def forward(self, input):
        return self.rms_norm(input.moveaxis(-1, -2)).moveaxis(-1, -2)
    
    
class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, transposed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.transposed = transposed
        self.buffer = None
        self.inference = False
        
    def forward(self, input):
        if self.transposed:
            input = input.moveaxis(-1, -2)
        
        kernel_size = self.kernel_size[0]
        if self.inference and self.buffer is None:
            self.buffer = torch.zeros(*input.shape[:-1], kernel_size - 1).to(input.device)
        
        if self.buffer is None:
            padding = kernel_size - 1
            input = F.pad(input, (padding, 0))
        else:
            input = torch.cat([self.buffer, input], dim=-1)
            self.buffer = input[..., -(kernel_size - 1):]
        
        output = super().forward(input)
        
        if self.transposed:
            output = output.moveaxis(-1, -2)
        return output


def get_norm(norm, num_features, ndim=2, random_init=False):
    match norm: 
        case 'batch':
            match ndim:
                case 1:
                    norm_layer = nn.BatchNorm1d(num_features)
                case 2:
                    norm_layer = nn.BatchNorm2d(num_features)
                case 3:
                    norm_layer = nn.BatchNorm3d(num_features)
                case _:
                    raise ValueError('Invalid dimensions.')
            
            if random_init:
                norm_layer.running_mean = torch.rand(num_features)
                norm_layer.running_var = torch.rand(num_features) + 1e-4
                norm_layer.weight.data = torch.rand(num_features)
                norm_layer.bias.data = torch.rand(num_features)
            
            return norm_layer
        case 'group':
            return nn.GroupNorm(4, num_features)
        case 'instance':
            return nn.GroupNorm(num_features, num_features)
        case 'layer':
            return nn.LayerNorm(num_features)
        case 'layer-feature':
            if num_features > 1:
                return LayerNormFeature(num_features)
            else:
                return nn.Identity()
        case 'rms':
            if num_features > 1:
                return RmsNormFeature(num_features)
            else:
                return nn.Identity()
        case None:
            return nn.Identity()
        case _:
            raise ValueError("Invalid normalization type.")


def get_postact(postact):
    if postact is None:
        return nn.Identity()
    
    postact_registry = {
        'relu': nn.ReLU(), 
        'relu6': nn.ReLU6(), 
        'lelu': nn.LeakyReLU(0.1), 
        'sigmoid': nn.Sigmoid(), 
        'tanh': nn.Tanh(), 
        'gelu': nn.GELU(), 
        'glu': nn.GLU(dim=1), 
        'silu': nn.SiLU(), 
    }
    
    if postact in postact_registry:
        return postact_registry[postact]
    else:
        raise ValueError("Invalid postact name.")            
    
    
def get_dropout(p, dropout_dim, num_features):
    if p is None:
        return nn.Identity()
    
    dropout_registry = {
        0: nn.Dropout,
        1: nn.Dropout1d,
        2: nn.Dropout2d,
        3: nn.Dropout3d,
    }
    
    if dropout_dim in dropout_registry:
        if dropout_dim == 0 or num_features >= 16:
            return dropout_registry[dropout_dim](p)
        else:
            return nn.Identity()
    else:
        raise ValueError("Invalid dimensions.")            
    

def get_activations(ndim, num_features, 
                    norm=None, postact=None, p=None, dropout_dim=0):
    if (norm is None) and (postact is None) and (p is None):
        return nn.Identity()
    
    activations = nn.Sequential()
    if norm is not None:
        activations.append(get_norm(norm, num_features, ndim))
    if postact is not None:
        activations.append(get_postact(postact))
    if p is not None:
        activations.append(get_dropout(p, dropout_dim, num_features))
    return activations
    
    
class LambdaLayer(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def pool_temporal(in_channels: int, 
                  out_channels: int, 
                  resample_ratio: int, 
                  resample_mode='subsampling', 
                  use_activations=True, 
                  ndim=1, 
                  **kwargs):
    if (resample_ratio is None) or (resample_ratio == 1):
        return nn.Identity()
    
    act_layer = get_activations(ndim=ndim, num_features=out_channels, **kwargs) if use_activations else nn.Identity()
    
    assert ndim == 1
    
    match resample_mode:
        case 'subsampling':
            assert in_channels == out_channels
            pool_layer = LambdaLayer(lambda input: input[..., (resample_ratio-1)::resample_ratio])
        
        case 'average':
            assert in_channels == out_channels
            pool_layer = Reduce('b c ... (t r) -> b c ... t', 'mean', r=resample_ratio)
            
        case 'repeat':
            assert in_channels == out_channels
            pool_layer = LambdaLayer(lambda x: einops.repeat(x, 'b c ... t -> b c ... (t r)', r=resample_ratio))
            
        case 'downsampling_dw':
            assert in_channels == out_channels
            pool_layer = EinMix('b c (t r) -> b c t', weight_shape='c r', c=in_channels, r=resample_ratio)
        
        case 'downsampling':
            pool_layer = nn.Conv1d(in_channels, out_channels, resample_ratio, resample_ratio, bias=False)
            
        case 'downsampling_with_bias':
            pool_layer = nn.Conv1d(in_channels, out_channels, resample_ratio, resample_ratio, bias=True)
            
        case 'upsampling':
            pool_layer = EinMix('b c t -> b d (t r)', weight_shape='c d r', c=in_channels, d=out_channels, r=resample_ratio)
            
        case 'upsampling_with_bias':
            pool_layer = EinMix('b c t -> b d (t r)', weight_shape='c d r', bias_shape='d', c=in_channels, d=out_channels, r=resample_ratio)
            
        case 'downshaping':
            assert in_channels * resample_ratio == out_channels
            pool_layer = Rearrange('b c (t r) -> b (c r) t', r=resample_ratio)
            
        case 'upshaping':
            assert out_channels * resample_ratio == in_channels
            pool_layer = Rearrange('b (c r) t -> b c (t r)', r=resample_ratio)
            
        case _:
            raise ValueError("Invalid downsampling mode.")
    
    if use_activations:
        return nn.Sequential(pool_layer, act_layer)
    else:
        return pool_layer


class ResidualConnect(nn.Module):
    def __init__(self, 
                 ndim, in_channels, out_channels, 
                 downshaping_ratio=None, offset=0, 
                 norm=None, postact=None, p=None, dropout_dim=0, 
                 connect_type=None, connect_kernel=1, transposed=False):
        super().__init__()
        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.connect_kernel = connect_kernel
        self.offset = offset
        self.connect_type = connect_type
        self.transposed = transposed
        
        self.norm_drop = nn.Sequential(
            get_norm(norm, out_channels, ndim), 
            get_dropout(p, dropout_dim=dropout_dim, num_features=out_channels)
        )
        self.postact_layer = get_postact(postact)
        
        if downshaping_ratio is not None:
            self.pool_layer = pool_temporal(in_channels, in_channels * downshaping_ratio, downshaping_ratio, 
                                            resample_mode='downshaping', ndim=ndim)
            in_channels = in_channels * downshaping_ratio
        else:
            self.pool_layer = nn.Identity()
        
        if connect_type != 'mix':
            assert in_channels == out_channels, f"{in_channels=} does not equal {out_channels=}"
        
        match connect_type:            
            case 'mix':
                self.skip_path = CausalConv1d(in_channels, out_channels, connect_kernel)
                
            case 'feedforward':
                assert ndim == 1
                self.skip_path = nn.Identity()
                self.mlp = nn.Sequential(                
                    nn.Conv1d(out_channels, out_channels, 1, bias=True), 
                    get_postact(postact), 
                    nn.Conv1d(out_channels, out_channels, 1, bias=False), 
                    get_activations(ndim, out_channels, norm, postact, p, dropout_dim)
                )
                
            case 'conv_module':
                assert ndim == 1
                expanded = 2 * out_channels
                self.skip_path = nn.Identity()
                def project(chin, chout, bias=True):
                    if self.transposed:
                        return nn.Linear(chin, chout, bias=bias)
                    else:
                        return nn.Conv1d(chin, chout, 1, bias=bias)
                self.mlp = nn.Sequential(
                    project(out_channels, expanded, bias=False), 
                    get_activations(ndim, expanded, norm, postact, p, dropout_dim), 
                    CausalConv1d(expanded, expanded, 4, transposed=self.transposed, groups=expanded), 
                    get_postact(postact), 
                    project(2 * out_channels, out_channels), 
                    get_dropout(p, dropout_dim, ndim), 
                )
                self.postact_module = get_postact(postact)
                
            case _:
                self.skip_path = nn.Identity()
    
    def forward(self, input, residual):        
        input = self.pool_layer(input)
        
        input = input[..., -residual.shape[-1]:]  # for causal conv
        input = self.skip_path(input)
        
        residual = self.norm_drop(residual)
        
        if self.connect_type == 'squeeze' and self.in_channels >= 16:
            padding = (0,) * 2 * (self.ndim - 1) + (2 * (self.connect_kernel - 1), 0)
            residual = self.se_block(residual, F.pad(input, padding))
            
        if self.offset > 0:
            input = F.pad(input[..., self.offset:], (0, self.offset))
        elif self.offset < 0:
            input = F.pad(input[..., :self.offset], (-self.offset, 0))
        
        match self.connect_type:
            case 'concat':
                output = self.postact_layer(torch.cat([input, residual], 1))
            case 'glu':
                output = self.postact_layer(input * torch.sigmoid(residual))
            case _:
                output = self.postact_layer(input + residual)
        
        if self.connect_type in ['feedforward', 'conv_module']:            
            output = output + self.mlp(output)
            if self.connect_type == 'conv_module':
                output = self.postact_module(output)
            
        return output


class ResidualConnectWithLayer(nn.Module):
    def __init__(self, 
                 residual_connect: ResidualConnect, 
                 layer: nn.Module):
        super().__init__()
        self.residual_connect = residual_connect
        self.layer = layer

    def forward(self, input):
        residual = self.layer(input)
        return self.residual_connect(input, residual)
        
        
def get_mlp(features: List[int], 
            norm=None, 
            postact='relu', 
            channel_first=False, 
            bias=True):
    """builds an MLP with the number of neurons for each layer
    specified by features
    """
    features = features
    
    if channel_first:
        layer = partial(nn.Conv1d, kernel_size=1, bias=bias)
    else:
        layer = partial(nn.Linear, bias=bias)
    
    block = nn.Sequential()
    for layer_id, (in_features, out_features) in enumerate(zip(features[:-1], features[1:])):
        block.append(layer(in_features, out_features))
        if layer_id < len(features) - 2:
            block.append(get_activations(ndim=1, num_features=out_features, norm=norm, postact=postact))
            
    return block
