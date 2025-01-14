from torch import nn

from centaurus.ssm import SSMLayer
from centaurus.utils import *


class Centaurus(nn.Module):
    """A generic Centaurus network (i.e. for classification)
    """
    def __init__(self, 
                 in_channels: int, 
                 num_classes: int, 
                 channels: int, 
                 depth: int, 
                 num_states=16, 
                 state_blocks=4, 
                 mode='s5', 
                 downsample=None, 
                 downsample_mode='downsampling', 
                 norm='batch', 
                 postact='relu', 
                 dropout=None, 
                 dropout_dim=1, 
                 skip=False, 
                 connect_type=None, 
                 connect_kernel=1, 
                 offset=0, 
                 **kwargs):
        super().__init__()    
        self.num_classes = num_classes
        
        self.num_states = make_list(num_states, depth)
        self.state_blocks = make_list(state_blocks, depth)
        self.modes = make_list(mode, depth)
        self.downsamples = make_list(downsample, depth, pad_mode='ones')
        self.downsamples = [1 if x is None else x for x in self.downsamples]
        self.norm = make_list(norm, depth)
        
        self.downsample_mode = downsample_mode
        self.postact = postact
        self.dropout = dropout
        self.dropout_dim = dropout_dim
        self.skips = make_list(skip, depth, repeat=True)
        self.connect_type = connect_type
        self.connect_kernel = connect_kernel
        self.offset = offset
        self.kwargs = kwargs
            
        self.channels = [in_channels] + make_list(channels, depth)
        
        self.backbone = nn.Sequential(
            *[self.ssm_block(c_in, c_out, coeffs, skip, downsample, state_blocks=b, mode=mode, norm=norm) 
              for (c_in, c_out, coeffs, skip, downsample, b, mode, norm) in 
              zip(self.channels[:-1], self.channels[1:], self.num_states, self.skips, self.downsamples, self.state_blocks, self.modes, self.norm)]
        )
        
        self.classifier = nn.Sequential(
            Reduce('b c ... -> b c', 'mean'), 
            get_mlp([self.channels[-1], self.channels[-1], num_classes], postact=postact, channel_first=False)
        )

    def ssm_layer(self, in_channels, out_channels, num_states, **kwargs):
        return SSMLayer(num_states, in_channels, out_channels, 
                        postact=self.postact, dropout=self.dropout, dropout_dim=self.dropout_dim, 
                        **self.kwargs, **kwargs)
        
    def ssm_block(self, 
                  in_channels, out_channels, num_states, 
                  skip=False, downsample=1, norm=None, 
                  **kwargs):
        pool_norm, pool_postact, pool_dropout = norm, self.postact, self.dropout
        
        match self.downsample_mode:
            case _ if self.downsample_mode.startswith('downsampling'):
                med_channels = in_channels
            case 'downshaping':
                med_channels = round(out_channels / downsample)
                pool_norm, pool_postact, pool_dropout = None, None, None
            case _:
                med_channels = out_channels
                pool_norm, pool_postact, pool_dropout = None, None, None
                
        if downsample == 1:
            med_channels = out_channels
        
        if (in_channels != med_channels) and (self.connect_type != 'mix'):
            skip = False
        
        ssm_layers = nn.Sequential()
        
        if skip:
            ssm_layer = self.ssm_layer(in_channels, med_channels, num_states, use_activations=False, **kwargs)
            residual_connect = ResidualConnect(1, in_channels, med_channels, offset=self.offset, 
                                               norm=norm, postact=self.postact, p=self.dropout, dropout_dim=self.dropout_dim, 
                                               connect_type=self.connect_type, connect_kernel=self.connect_kernel)
            ssm_layers.append(ResidualConnectWithLayer(residual_connect, ssm_layer))
        else:            
            ssm_layers.append(self.ssm_layer(in_channels, med_channels, num_states, use_activations=True, norm=norm, **kwargs))
        
        if self.downsample_mode is not None:
            ssm_layers.append(pool_temporal(med_channels, out_channels, downsample, self.downsample_mode, 
                                            norm=pool_norm, postact=pool_postact, p=pool_dropout, dropout_dim=self.dropout_dim))
        
        return ssm_layers
    
    def forward(self, input):
        return self.classifier(self.backbone(input))
    
    def set_run_mode(self, run_mode='training'):
        for layer in self.modules():
            if isinstance(layer, SSMLayer):
                layer.set_run_mode(run_mode)
                
    def reset_states(self):
        for layer in self.modules():
            if isinstance(layer, SSMLayer):
                layer.reset_state()


class CentaurusSE(Centaurus):
    """Centaurus autoencoder for speech enhancement
    """
    def __init__(self, 
                 in_channels: int, 
                 channels: int, 
                 num_states=16, 
                 state_blocks=4, 
                 mode='s5', 
                 downsample=None, 
                 downsample_mode='downsampling', 
                 norm='layer-feature', 
                 postact='silu', 
                 **kwargs):
        nn.Module.__init__(self)
        depth = len(channels)
        self.depth = depth
        self.channels = [in_channels] + channels
        self.num_states = make_list(num_states, depth)
        self.state_blocks = make_list(state_blocks, depth)
        self.modes = make_list(mode, depth)
        
        self.downsamples = downsample
        self.norm = norm
        self.postact = postact
        self.kwargs = kwargs
        
        self.downsample_mode = downsample_mode
        if downsample_mode == 'downshaping':
            med_channels = [round(channel / downsample) for (channel, downsample) in zip(self.channels[1:], self.downsamples)]
        elif downsample_mode.startswith('downsampling'):
            med_channels = self.channels[:-1]
        elif downsample_mode == 'average':
            med_channels = self.channels[1:]
        else:
            raise ValueError("Invalid downsample mode.")
        
        self.down_ssms = nn.ModuleList([self.ssm_pool(n, c_in, c_med, c_out, r, state_blocks=b, mode=mode, ssm_first=True) 
                                        for (n, c_in, c_med, c_out, mode, r, b) in 
                                        zip(self.num_states, self.channels[:-1], med_channels, self.channels[1:], self.modes, self.downsamples, self.state_blocks)])
        self.up_ssms = nn.ModuleList([self.ssm_pool(n, c_in, c_med, c_out, r, state_blocks=b, mode=mode, ssm_first=False) 
                                      for (n, c_in, c_med, c_out, mode, r, b) in 
                                      zip(self.num_states, self.channels[1:], med_channels, self.channels[:-1], self.modes, self.downsamples, self.state_blocks)])
        
        self.hid_ssms = self.ssm_block(self.num_states[-1], self.channels[-1], self.channels[-1], mode='s5', 
                                       state_blocks=self.state_blocks[-1], use_activations=True, block_repeat=2)
        self.last_ssms = self.ssm_block(self.num_states[0], self.channels[0], self.channels[0], mode=self.modes[0], 
                                        state_blocks=self.state_blocks[0], use_activations=False, block_repeat=2)
    
    def ssm_pool(self, num_states, in_channels, med_channels, out_channels, 
                 downsample, ssm_first=True, **kwargs):
        upsample_mode = 'up' + self.downsample_mode[4:]
        if self.downsample_mode == 'average':
            upsample_mode = 'repeat'
        
        if ssm_first:
            ssm_blocks = nn.Sequential(
                self.ssm_block(num_states, in_channels, med_channels, use_activations=True, **kwargs), 
                self.pool_layer(med_channels, out_channels, downsample, self.downsample_mode), 
            )
        else:
            ssm_blocks = nn.Sequential(                
                self.pool_layer(in_channels, med_channels, downsample, upsample_mode), 
                self.ssm_block(num_states, med_channels, out_channels, use_activations=True, **kwargs), 
            )
        
        return ssm_blocks
    
    def ssm_layer(self, *args, **kwargs):
        return SSMLayer(*args, norm=self.norm, postact=self.postact, **kwargs, **self.kwargs)
    
    def pool_layer(self, chin, chout, ratio, mode):
        return pool_temporal(chin, chout, ratio, mode, use_activations=False)
    
    def ssm_block(self, *args, block_repeat=1, use_activations=True, **kwargs):
        layers = nn.Sequential()
        
        for _ in range(block_repeat - 1):
            layers.append(self.ssm_layer(*args, use_activations=True, **kwargs))
        layers.append(self.ssm_layer(*args, use_activations=use_activations, **kwargs))
            
        return layers
    
    def forward(self, input):
        x, skips = input, []
        
        # encoder
        for ssm in self.down_ssms:
            if self.downsample_mode in ['downshaping', 'average']:
                x = ssm[0](x)
                skips.append(x)
                x = ssm[1](x)
            else:
                skips.append(x)
                x = ssm(x)
        
        x = self.hid_ssms(x)
        
        # decoder
        for (ssm, skip) in zip(self.up_ssms[::-1], skips[::-1]):
            x = ssm[0](x)
            x = x + skip
            x = ssm[1](x)
            
        return self.last_ssms(x)
