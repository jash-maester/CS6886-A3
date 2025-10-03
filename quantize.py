import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# ---------- Core Uniform Quantization Helpers ----------
def qparams_from_minmax(xmin, xmax, n_bits=8, unsigned=False, eps=1e-12):
    """
    Calculate quantization parameters (scale, zero_point, qmin, qmax).
    
    Args:
        xmin: Minimum value in tensor
        xmax: Maximum value in tensor
        n_bits: Number of bits for quantization
        unsigned: If True, uses unsigned range [0, 2^n-1], else symmetric
        eps: Small value to prevent division by zero
    
    Returns:
        scale, zero_point, qmin, qmax
    """
    if unsigned:
        qmin, qmax = 0, (1 << n_bits) - 1
        xmin = torch.zeros_like(xmin)  # For post-ReLU, min is 0
        scale = (xmax - xmin).clamp_min(eps) / float(qmax - qmin)
        zp = torch.round(-xmin / scale).clamp(qmin, qmax)
    else:
        # Symmetric quantization for weights
        qmax = (1 << (n_bits - 1)) - 1
        qmin = -qmax
        max_abs = torch.max(xmin.abs(), xmax.abs()).clamp_min(eps)
        scale = max_abs / float(qmax)
        zp = torch.zeros_like(scale)
    return scale, zp, int(qmin), int(qmax)

def quantize(x, scale, zp, qmin, qmax):
    """Quantize tensor x using scale and zero_point."""
    q = torch.round(x / scale + zp)
    return q.clamp(qmin, qmax)

def dequantize(q, scale, zp):
    """Dequantize tensor q using scale and zero_point."""
    return (q - zp) * scale

# ---------- Activation Fake Quantization ----------
class ActFakeQuant(nn.Module):
    """
    Per-tensor activation fake quantization with calibration phase.
    Used after ReLU activations with unsigned quantization.
    """
    def __init__(self, n_bits=8, unsigned=True):
        super().__init__()
        self.n_bits = n_bits
        self.unsigned = unsigned
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zp", torch.tensor(0.0))
        self.frozen = False
        self.qmin, self.qmax = None, None

    @torch.no_grad()
    def observe(self, x):
        """Observe min/max values during calibration."""
        self.min_val = torch.minimum(self.min_val, x.min())
        self.max_val = torch.maximum(self.max_val, x.max())

    @torch.no_grad()
    def freeze(self):
        """Compute final quantization parameters after calibration."""
        scale, zp, qmin, qmax = qparams_from_minmax(
            self.min_val, self.max_val, n_bits=self.n_bits, unsigned=self.unsigned
        )
        self.scale.copy_(scale)
        self.zp.copy_(zp)
        self.qmin, self.qmax = qmin, qmax
        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            # Calibration phase: observe values
            self.observe(x)
            return x
        # Quantization phase: fake quantize
        q = quantize(x, self.scale, self.zp, self.qmin, self.qmax)
        return dequantize(q, self.scale, self.zp)

# ---------- Weight Quantization Wrappers ----------
class QuantConv2d(nn.Conv2d):
    """
    Conv2d layer with per-tensor symmetric weight quantization.
    """
    def __init__(self, *args, weight_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.register_buffer("w_scale", torch.tensor(1.0))
        self.register_buffer("w_zp", torch.tensor(0.0))
        self.frozen = False
        self.qmin = None
        self.qmax = None

    @torch.no_grad()
    def freeze(self):
        """Compute quantization parameters from trained weights."""
        w = self.weight.detach()
        w_min, w_max = w.min(), w.max()
        scale, zp, qmin, qmax = qparams_from_minmax(
            w_min, w_max, n_bits=self.weight_bits, unsigned=False
        )
        self.w_scale.copy_(scale)
        self.w_zp.copy_(zp)
        self.qmin, self.qmax = qmin, qmax
        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            return F.conv2d(x, self.weight, self.bias, self.stride, 
                          self.padding, self.dilation, self.groups)
        # Fake quantize weights
        q = quantize(self.weight, self.w_scale, self.w_zp, self.qmin, self.qmax)
        w_dq = dequantize(q, self.w_scale, self.w_zp)
        return F.conv2d(x, w_dq, self.bias, self.stride, 
                       self.padding, self.dilation, self.groups)

class QuantLinear(nn.Linear):
    """
    Linear layer with per-tensor symmetric weight quantization.
    """
    def __init__(self, *args, weight_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.register_buffer("w_scale", torch.tensor(1.0))
        self.register_buffer("w_zp", torch.tensor(0.0))
        self.frozen = False
        self.qmin = None
        self.qmax = None

    @torch.no_grad()
    def freeze(self):
        """Compute quantization parameters from trained weights."""
        w = self.weight.detach()
        w_min, w_max = w.min(), w.max()
        scale, zp, qmin, qmax = qparams_from_minmax(
            w_min, w_max, n_bits=self.weight_bits, unsigned=False
        )
        self.w_scale.copy_(scale)
        self.w_zp.copy_(zp)
        self.qmin, self.qmax = qmin, qmax
        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            return F.linear(x, self.weight, self.bias)
        # Fake quantize weights
        q = quantize(self.weight, self.w_scale, self.w_zp, self.qmin, self.qmax)
        w_dq = dequantize(q, self.w_scale, self.w_zp)
        return F.linear(x, w_dq, self.bias)

# ---------- Model Surgery ----------
def swap_to_quant_modules(model, weight_bits=8, act_bits=8, activations_unsigned=True):
    """
    Replace standard layers with quantized versions.
    - Conv2d/Linear -> QuantConv2d/QuantLinear
    - ReLU/ReLU6 -> Sequential(ReLU, ActFakeQuant)
    """
    for name, m in list(model.named_children()):
        # Recursively process child modules
        swap_to_quant_modules(m, weight_bits, act_bits, activations_unsigned)

        # Replace Conv2d with QuantConv2d
        if isinstance(m, nn.Conv2d):
            q = QuantConv2d(
                m.in_channels, m.out_channels, m.kernel_size,
                stride=m.stride, padding=m.padding, dilation=m.dilation,
                groups=m.groups, bias=(m.bias is not None),
                weight_bits=weight_bits
            )
            q.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                q.bias.data.copy_(m.bias.data)
            setattr(model, name, q)

        # Replace Linear with QuantLinear
        elif isinstance(m, nn.Linear):
            q = QuantLinear(
                m.in_features, m.out_features, 
                bias=(m.bias is not None), 
                weight_bits=weight_bits
            )
            q.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                q.bias.data.copy_(m.bias.data)
            setattr(model, name, q)

        # Replace ReLU with ReLU + ActFakeQuant
        elif isinstance(m, (nn.ReLU, nn.ReLU6)):
            is_relu6 = isinstance(m, nn.ReLU6)
            inplace = getattr(m, "inplace", False)
            
            if is_relu6:
                relu_layer = nn.ReLU6(inplace=inplace)
            else:
                relu_layer = nn.ReLU(inplace=inplace)
            
            seq = nn.Sequential(OrderedDict([
                ("relu", relu_layer),
                ("aq", ActFakeQuant(n_bits=act_bits, unsigned=activations_unsigned)),
            ]))
            setattr(model, name, seq)

def freeze_all_quant(model):
    """
    Freeze all quantization parameters after calibration.
    """
    for mod in model.modules():
        if isinstance(mod, (QuantConv2d, QuantLinear)):
            mod.freeze()
        if isinstance(mod, nn.Sequential):
            for sub in mod.modules():
                if isinstance(sub, ActFakeQuant):
                    sub.freeze()

# ---------- Size Calculation Utilities ----------
def model_size_bytes_fp32(model):
    """Calculate total model size if all parameters are FP32."""
    total = 0
    for p in model.parameters():
        total += p.numel() * 4  # 4 bytes per FP32
    return total

def model_size_bytes_quant(model, weight_bits=8):
    """
    Calculate model size with quantized weights.
    Weights stored as intN, biases kept as FP32.
    """
    total = 0
    for name, p in model.named_parameters():
        if "weight" in name:
            total += p.numel() * weight_bits // 8
        elif "bias" in name:
            total += p.numel() * 4  # Biases stay FP32
    return total

def calculate_metadata_size(model):
    """
    Calculate overhead from quantization metadata (scales and zero points).
    Each quantized layer has scale (FP32) and zero_point (FP32).
    """
    metadata_bytes = 0
    for mod in model.modules():
        if isinstance(mod, (QuantConv2d, QuantLinear)):
            metadata_bytes += 8  # scale (4 bytes) + zero_point (4 bytes)
        if isinstance(mod, ActFakeQuant):
            metadata_bytes += 8  # scale + zero_point
    return metadata_bytes

def print_compression(model, weight_bits=8, act_bits=8):
    """Print detailed compression statistics."""
    fp32_size = model_size_bytes_fp32(model)
    quant_size = model_size_bytes_quant(model, weight_bits)
    metadata_size = calculate_metadata_size(model)
    total_size = quant_size + metadata_size
    
    # Weight-only compression
    total_weights = sum(p.numel() for name, p in model.named_parameters() if "weight" in name)
    weight_fp32_size = total_weights * 4
    weight_quant_size = total_weights * weight_bits // 8
    weight_compression = weight_fp32_size / max(weight_quant_size, 1)
    
    # Overall compression
    overall_compression = fp32_size / max(total_size, 1)
    
    print("\n" + "="*60)
    print("COMPRESSION SUMMARY")
    print("="*60)
    print(f"FP32 Model Size:           {fp32_size/1024/1024:.2f} MB")
    print(f"Quantized Weights:         {quant_size/1024/1024:.2f} MB ({weight_bits}-bit)")
    print(f"Metadata Overhead:         {metadata_size/1024:.2f} KB")
    print(f"Total Quantized Size:      {total_size/1024/1024:.2f} MB")
    print(f"-"*60)
    print(f"Weight Compression Ratio:  {weight_compression:.2f}x")
    print(f"Overall Compression Ratio: {overall_compression:.2f}x")
    print(f"Model Size Reduction:      {100*(1 - total_size/fp32_size):.1f}%")
    print(f"-"*60)
    print(f"Weight Quantization:       {weight_bits} bits")
    print(f"Activation Quantization:   {act_bits} bits")
    print("="*60 + "\n")
    
    return {
        'fp32_size_mb': fp32_size/1024/1024,
        'quant_size_mb': total_size/1024/1024,
        'weight_compression': weight_compression,
        'overall_compression': overall_compression,
        'metadata_kb': metadata_size/1024
    }
