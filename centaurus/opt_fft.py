import torch
from torch.amp import custom_bwd, custom_fwd


@torch.compile
def _K(dtA_real, dtA_imag, length, weight=None, dim=-2, complex_proj=False, l_shift=0):
    device = dtA_real.device
    lrange = torch.arange(l_shift, length + l_shift, device=device)
    
    with torch.autocast('cuda', enabled=False):
        dtA_real, dtA_imag = dtA_real.float(), dtA_imag.float()
        if complex_proj:
            K = (torch.complex(dtA_real, dtA_imag)[..., None] * lrange).exp()
        else:
            K = (dtA_real[..., None] * lrange).exp() * torch.cos(dtA_imag[..., None] * lrange)
            
        if weight is not None:
            return (K * weight[..., None]).sum(dim)
        else:
            return K


def _full_k(dtA_real, dtA_imag, B, C, E, length):
    K = _K(dtA_real, dtA_imag, length, weight=E)
    return (B[..., None] * C[..., None, None] * K[:, None, :]).sum(1)


class PaddedFFTConv(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable
    @custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, u, k, n, mode, is_complex=False):        
        if is_complex:
            uf = torch.fft.fft(u, 2*n)
            kf = torch.fft.fft(k, 2*n)
        else:
            uf = torch.fft.rfft(u, 2*n)
            kf = torch.fft.rfft(k, 2*n)
        
        if mode == 'dw':
            yf = uf * kf
        elif mode == 'full':
            yf = torch.einsum('bcl,dcl->bdl', uf, kf)
        
        ctx.is_complex = is_complex
        ctx.mode = mode
        ctx.n = n
        
        ctx.save_for_backward(u, k)
        
        if is_complex:
            return torch.fft.ifft(yf)[..., :n]
        else:
            return torch.fft.irfft(yf)[..., :n]
    
    @staticmethod
    @torch.compiler.disable
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        is_complex = ctx.is_complex
        mode = ctx.mode
        n = ctx.n
        u, k = ctx.saved_tensors
        
        if is_complex:
            uf = torch.fft.fft(u, 2*n)
            kf = torch.fft.fft(k, 2*n)
            grad_yf = torch.fft.fft(grad_output, 2*n)
        else:
            uf = torch.fft.rfft(u, 2*n)
            kf = torch.fft.rfft(k, 2*n)
            grad_yf = torch.fft.rfft(grad_output, 2*n)
        
        if mode == 'dw':
            grad_uf = grad_yf * torch.conj(kf)
        elif mode == 'full':
            grad_uf = torch.einsum('bdl,dcl->bcl', grad_yf, torch.conj(kf))
            
        if is_complex:
            grad_u = torch.fft.ifft(grad_uf, 2*n)[..., :n]
        else:
            grad_u = torch.fft.irfft(grad_uf, 2*n)[..., :n]

        if mode == 'dw':
            grad_kf = torch.einsum('bnl,bnl->nl', grad_yf, torch.conj(uf))
        elif mode == 'full':
            grad_kf = torch.einsum('bdl,bcl->dcl', grad_yf, torch.conj(uf))
        
        if is_complex:    
            grad_k = torch.fft.ifft(grad_kf, 2*n)[..., :n]
        else:
            grad_k = torch.fft.irfft(grad_kf, 2*n)[..., :n]
        
        return grad_u, grad_k, None, None, None, None, None
    

def padded_fft_conv_opt(input, dtA_real, dtA_imag, B, C, E):
    batch, chin, length = input.shape
    if B is None:
        K = _K(dtA_real, dtA_imag, length, weight=E)
        if K.ndim == 3:
            return PaddedFFTConv.apply(input, K, length, 'full')
        elif K.ndim == 2:
            return PaddedFFTConv.apply(input, K, length, 'dw')
    
    chout, coeffs = C.shape
    
    if (1 / chin + 1 / chout) > (1 / batch + 1 / coeffs):
        kernel = _full_k(dtA_real, dtA_imag, B, C, E, length)
        return PaddedFFTConv.apply(input, kernel, length, 'full', False)
    else:
        K = _K(dtA_real, dtA_imag, length, weight=E)
        x = torch.einsum('bcl,nc->bnl', input, B)
        x = PaddedFFTConv.apply(x, K, length, 'dw', False)
        return torch.einsum('bnl,dn->bdl', x, C)
    
    
class PaddedFFTConvInference(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.K = None
        
    def forward(self, input, dtA_real, dtA_imag, B, C, E):
        input_len = input.shape[-1]
        if (self.K is None) or (input_len > self.K.shape[-1]):
            self.K = _K(dtA_real, dtA_imag, input_len, weight=E).detach()
            print(f"cached kernel of shape {self.K.shape}")
        K_trunc = self.K[..., :input_len] if input_len < self.K.shape[-1] else self.K
        
        if B is None:
            if self.K.ndim == 3:
                return PaddedFFTConv.apply(input, K_trunc, input_len, 'full')
            elif self.K.ndim == 2:
                return PaddedFFTConv.apply(input, K_trunc, input_len, 'dw')
            
        x = torch.einsum('bcl,nc->bnl', input, B)
        x = PaddedFFTConv.apply(x, K_trunc, input_len, 'dw', False)
        return torch.einsum('bnl,dn->bdl', x, C)
