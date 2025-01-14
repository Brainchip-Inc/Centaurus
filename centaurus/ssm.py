import math

import einops
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from centaurus.opt_fft import _K, PaddedFFTConvInference, padded_fft_conv_opt
from centaurus.utils import *


class Kernelizer(nn.Module):
    """The core module for performing SSM operations
    """
    def __init__(self, mode='s5', **kwargs):
        """
        Args:
            mode (str, optional): The mode of the Centaurus layer (connectivity).
                Available modes are ['dws', 's5', 'neck', 'full'].
        """
        super().__init__()
        self.mode = mode
        
        self.set_run_mode('training')
        self.state = None        

    @torch.compiler.disable
    def discretize(self, 
                   A: torch.Tensor, 
                   weight: torch.Tensor, 
                   log_dt: torch.Tensor):   
        """
        Generates discretized input and state matrices using a simplified 
        zero-order-hold method. Assumes that the state matrix is diagonal.

        Args:
            A (torch.Tensor): the diagonal state matrix
            weight (torch.Tensor): the weighting tensor (or the generalized B matrix)
            log_dt (torch.Tensor): the log of the timestep log_dt

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): The discretized:
                - real part of the state matrix
                - imaginary part of the state matrix
                - the (real) weight tensor
        """
        with torch.autocast('cuda', enabled=False):
            A_real, A_imag = -F.softplus(A[..., 0]), A[..., 1]
            dt = log_dt.exp()
        
            match self.mode:
                case 'neck':
                    dt = dt.unsqueeze(-1)  # (N, :)
                    weight_hat = weight * dt
                case 'full':
                    dt = dt.unsqueeze(-2)  # (D, :, C)
                    weight_hat = weight * dt
                case 'dws':
                    weight_hat = weight * dt  # (C, N)
                case _:
                    weight_hat = weight * dt.unsqueeze(-1)  # (N, :)

            dtA_real, dtA_imag = dt * A_real, dt * A_imag
            return dtA_real, dtA_imag, weight_hat
    
    def forward(self, 
                input: torch.Tensor, 
                A: torch.Tensor, 
                B: torch.Tensor, 
                C: torch.Tensor, 
                log_dt: torch.Tensor, 
                E: torch.tensor):
        match self.mode:
            case 's5' | 'neck':
                dtA_real, dtA_imag, B_hat = self.discretize(A, B, log_dt)
                if self.run_mode != 'inference':
                    return padded_fft_conv_opt(input, dtA_real, dtA_imag, B_hat, C, E)
                y_intra = self.fft_inference(input, dtA_real, dtA_imag, B_hat, C, E)

            case 'dws' | 'full':
                dtA_real, dtA_imag, E_hat = self.discretize(A, E, log_dt)
                if self.run_mode != 'inference':
                    return padded_fft_conv_opt(input, dtA_real, dtA_imag, None, None, E_hat)
                y_intra = self.fft_inference(input, dtA_real, dtA_imag, None, None, E_hat)
        
        K_flip = _K(dtA_real, dtA_imag, input.shape[-1], complex_proj=True).flip(-1)
        K_advance = _K(dtA_real, dtA_imag, input.shape[-1], complex_proj=True, l_shift=1)
        k = K_advance[..., -1]
        
        match self.mode:
            case 'dws':
                self.state = torch.zeros(input.shape[0], *E.shape).to(input.device).cfloat() if self.state is None else self.state
                y_decay = torch.einsum('bcn,cn,cnl->bcl', self.state, E_hat.cfloat(), K_advance).real
                self.state = self.state * k + torch.einsum('bcl,cnl->bcn', input.cfloat(), K_flip)
                return y_intra + y_decay
            case 'full':
                self.state = torch.zeros(input.shape[0], *E.shape).to(input.device).cfloat() if self.state is None else self.state
                y_decay = torch.einsum('bdcn,dcn,dcnl->bdl', self.state, E_hat.cfloat(), K_advance).real
                self.state = self.state * k + torch.einsum('bcl,dcnl->bdcn', input.cfloat(), K_flip)
                return y_intra + y_decay
            case 's5':
                self.state = torch.zeros(input.shape[0], B_hat.shape[0]).to(input.device).cfloat() if self.state is None else self.state
                y_decay = torch.einsum('bn,nl,dn->bdl', self.state, K_advance, C.cfloat()).real
                self.state = self.state * k + torch.einsum('bcl,nc,nl->bn', input.cfloat(), B_hat.cfloat(), K_flip)
                return y_intra + y_decay
            case 'neck':
                self.state = torch.zeros(input.shape[0], B_hat.shape[0], E.shape[-1]).to(input.device).cfloat() if self.state is None else self.state
                y_decay = torch.einsum('bnm,nml,dn->bdl', self.state, K_advance, C.cfloat()).real
                self.state = self.state * k + torch.einsum('bcl,nc,nml,nm->bnm', input.cfloat(), B_hat.cfloat(), K_flip, E.cfloat())
                return y_intra + y_decay
        
    def set_run_mode(self, run_mode='training'):
        if run_mode not in ['training', 'inference']:
            raise ValueError("Invalid run mode.")
        self.run_mode = run_mode
        if run_mode == 'inference':
            self.fft_inference = PaddedFFTConvInference()

    def reset_state(self):
        self.state = None


class SSMLayer(Kernelizer):
    """An augmentation of the `Kernelizer` class, 
    that handles the SSM weight initialization
    """
    def __init__(self, 
                 num_states: int, 
                 in_channels: int, 
                 out_channels: int, 
                 state_blocks=None, 
                 norm='batch', 
                 postact='relu', 
                 dropout=None, 
                 dropout_dim=1, 
                 use_activations=False, 
                 **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.state_blocks = 1 if state_blocks is None else state_blocks
        
        self.norm = norm
        self.postact = postact
        self.dropout = dropout
        self.dropout_dim = dropout_dim
        
        self.E = None
        
        inv_softplus = lambda x: x + np.log(-np.expm1(-x))
        A = np.stack([0.5 * np.ones(num_states), math.pi * np.arange(num_states)], -1)
        A[..., 0] = inv_softplus(A[..., 0])
        
        if self.mode in ['dws']:
            dt = np.geomspace(1e-3, 1e-1, in_channels)
        elif self.mode == 'full':
            dt = np.geomspace(1e-3, 1e-1, out_channels)
        else:
            dt = np.geomspace(1e-3, 1e-1, self.state_blocks)
            
        log_dt = np.log(dt)
        
        def to_parameter(mat, is_complex=False, requires_grad=True):
            if isinstance(mat, list):
                return [to_parameter(m, is_complex, requires_grad) for m in mat]
            if mat is None:
                return None
            tensor = torch.tensor(mat, dtype=torch.float)
            if is_complex:
                tensor = tensor.cfloat()
            return Parameter(tensor, requires_grad=requires_grad)
        
        def ones(shape, fan_in):
            mat = np.ones(shape) / math.sqrt(fan_in)
            return to_parameter(mat)
        
        def normal(shape, fan_in):
            mat = np.random.randn(*shape) * math.sqrt(2 / fan_in)
            return to_parameter(mat)
        
        tot_coeffs = self.state_blocks * num_states
        match self.mode:
            case 'dws':
                log_dt = einops.repeat(log_dt, 'c -> c n', n=num_states)
                A = einops.repeat(A, 'n i -> c n i', c=in_channels)
                self.B = None
                self.C = None
                self.E = ones((in_channels, num_states), num_states)
            
            case 's5':
                log_dt = einops.repeat(log_dt, 'j -> (j n)', n=num_states)
                A = einops.repeat(A, 'n i -> (j n) i', j=self.state_blocks)
                self.B = ones((tot_coeffs, in_channels), in_channels)
                self.C = normal((out_channels, tot_coeffs), tot_coeffs)
                self.E = None

            case 'neck':
                # does not require log_dt repeating
                A = einops.repeat(A, 'n i -> r n i', r=self.state_blocks)                    
                self.B = ones((self.state_blocks, in_channels), in_channels)
                self.C = normal((out_channels, self.state_blocks), tot_coeffs)
                self.E = normal((self.state_blocks, num_states), 1)
                
            case 'full':
                log_dt = einops.repeat(log_dt, 'd -> d n', n=num_states)
                A = einops.repeat(A, 'n i -> d c n i', c=in_channels, d=out_channels)
                self.B = None
                self.C = None
                self.E = ones((out_channels, in_channels, num_states), in_channels)
        
        self.A = to_parameter(A)
        self.log_dt = to_parameter(log_dt)
        self._register_sensitives(self.log_dt, self.A)
        
        if self.mode in ['dws']:
            self.mixer = nn.Sequential(
                self.act_layer(in_channels), 
                nn.Conv1d(in_channels, out_channels, 1, bias=False), 
                self.act_layer(out_channels, use_activations=use_activations)
            )
        else:
            self.mixer = self.act_layer(out_channels) if use_activations else nn.Identity()
    
    @staticmethod
    def _register_sensitives(*args):
        """Registers certain parameters as "sensitive", potentially signaling 
        the optimizer to apply smaller lr to them
        """
        for arg in args:
            if isinstance(arg, nn.Module):
                for param in arg.parameters():
                    param.sensitive = True
                continue
            arg.sensitive = True
    
    def act_layer(self, num_features, use_activations=True):
        if use_activations:
            return get_activations(1, num_features, self.norm, self.postact, self.dropout, self.dropout_dim)
        else:
            return nn.Identity()
    
    def forward(self, input):
        return self.mixer(super().forward(input, self.A, self.B, self.C, self.log_dt, E=self.E))
        