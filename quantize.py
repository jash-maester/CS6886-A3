import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

# ---------- Advanced Quantization Helpers ----------
def qparams_from_minmax(xmin, xmax, n_bits=8, unsigned=False, eps=1e-12, symmetric=False):
    """
    Enhanced quantization parameter calculation with symmetric option.

    Args:
        xmin: Minimum value in tensor
        xmax: Maximum value in tensor
        n_bits: Number of bits for quantization
        unsigned: If True, uses unsigned range [0, 2^n-1]
        eps: Small value to prevent division by zero
        symmetric: Force symmetric quantization around zero

    Returns:
        scale, zero_point, qmin, qmax
    """
    if unsigned:
        qmin, qmax = 0, (1 << n_bits) - 1
        xmin = torch.zeros_like(xmin)  # For post-ReLU, min is 0
        scale = (xmax - xmin).clamp_min(eps) / float(qmax - qmin)
        zp = torch.round(-xmin / scale).clamp(qmin, qmax)
    elif symmetric:
        # Enhanced symmetric quantization
        qmax = (1 << (n_bits - 1)) - 1
        qmin = -qmax - 1 if n_bits == 8 else -qmax  # Allow -128 for 8-bit symmetric
        max_abs = torch.max(xmin.abs(), xmax.abs()).clamp_min(eps)
        scale = max_abs / float(qmax)
        zp = torch.zeros_like(scale)
    else:
        # Asymmetric signed quantization
        qmax = (1 << (n_bits - 1)) - 1
        qmin = -qmax - 1 if n_bits == 8 else -qmax

        # Ensure range includes zero for better numerical stability
        xmin = torch.min(xmin, torch.zeros_like(xmin))
        xmax = torch.max(xmax, torch.zeros_like(xmax))

        scale = (xmax - xmin).clamp_min(eps) / float(qmax - qmin)
        zp = torch.round(qmin - xmin / scale).clamp(qmin, qmax)

    return scale, zp, int(qmin), int(qmax)

def quantize(x, scale, zp, qmin, qmax):
    """Quantize tensor x using scale and zero_point with gradient straight-through."""
    q = torch.round(x / scale + zp)
    q = q.clamp(qmin, qmax)
    return q

def dequantize(q, scale, zp):
    """Dequantize tensor q using scale and zero_point."""
    return (q - zp) * scale

def fake_quantize(x, scale, zp, qmin, qmax):
    """Fake quantization with straight-through estimator."""
    q = quantize(x, scale, zp, qmin, qmax)
    return dequantize(q, scale, zp)

# ---------- Advanced Activation Quantization ----------
class AdvancedActFakeQuant(nn.Module):
    """
    Advanced activation fake quantization with multiple calibration methods.
    """
    def __init__(self, n_bits=8, unsigned=True, calibration='minmax', percentile=99.9):
        super().__init__()
        self.n_bits = n_bits
        self.unsigned = unsigned
        self.calibration = calibration
        self.percentile = percentile

        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zp", torch.tensor(0.0))
        self.frozen = False
        self.qmin, self.qmax = None, None

        # For percentile calibration
        self.register_buffer("histogram", None)
        self.register_buffer("hist_min", torch.tensor(float("inf")))
        self.register_buffer("hist_max", torch.tensor(float("-inf")))

    @torch.no_grad()
    def observe(self, x):
        """Observe values during calibration with different methods."""
        if self.calibration == 'minmax':
            self.min_val = torch.minimum(self.min_val, x.min())
            self.max_val = torch.maximum(self.max_val, x.max())
        elif self.calibration == 'moving_average':
            # Exponential moving average of min/max
            alpha = 0.01
            self.min_val = alpha * x.min() + (1 - alpha) * self.min_val
            self.max_val = alpha * x.max() + (1 - alpha) * self.max_val
        elif self.calibration == 'percentile':
            # Update histogram for percentile-based calibration
            if self.histogram is None:
                self.hist_min = x.min()
                self.hist_max = x.max()
                self.histogram = torch.histc(x, bins=2048, min=self.hist_min, max=self.hist_max)
            else:
                # Update histogram range if needed
                new_min = torch.minimum(self.hist_min, x.min())
                new_max = torch.maximum(self.hist_max, x.max())

                if new_min != self.hist_min or new_max != self.hist_max:
                    # Rebin histogram (simplified approach)
                    current_bins = 2048
                    self.histogram = torch.histc(x, bins=current_bins, min=new_min, max=new_max)
                    self.hist_min = new_min
                    self.hist_max = new_max
                else:
                    # Add to existing histogram
                    self.histogram += torch.histc(x, bins=2048, min=self.hist_min, max=self.hist_max)

    @torch.no_grad()
    def freeze(self):
        """Compute final quantization parameters after calibration."""
        if self.calibration == 'percentile' and self.histogram is not None:
            # Percentile-based range estimation
            total = self.histogram.sum()
            if total > 0:
                cdf = torch.cumsum(self.histogram, dim=0) / total
                lower_idx = torch.searchsorted(cdf, (100 - self.percentile) / 200.0)
                upper_idx = torch.searchsorted(cdf, 1 - (100 - self.percentile) / 200.0)

                bin_width = (self.hist_max - self.hist_min) / len(self.histogram)
                xmin = self.hist_min + lower_idx * bin_width
                xmax = self.hist_min + upper_idx * bin_width
            else:
                xmin, xmax = self.min_val, self.max_val
        else:
            xmin, xmax = self.min_val, self.max_val

        scale, zp, qmin, qmax = qparams_from_minmax(
            xmin, xmax, n_bits=self.n_bits, unsigned=self.unsigned
        )
        self.scale.copy_(scale)
        self.zp.copy_(zp)
        self.qmin, self.qmax = qmin, qmax
        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            self.observe(x)
            return x

        # Use fake quantization in forward pass
        if self.training:
            # Add quantization noise during training for better QAT
            noise = (torch.rand_like(x) - 0.5) * self.scale
            x = x + noise

        return fake_quantize(x, self.scale, self.zp, self.qmin, self.qmax)

# ---------- Advanced Weight Quantization ----------
class AdvancedQuantConv2d(nn.Conv2d):
    """
    Enhanced Conv2d with per-channel or per-tensor weight quantization.
    """
    def __init__(self, *args, weight_bits=8, per_channel=False, symmetric=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.per_channel = per_channel
        self.symmetric = symmetric

        if per_channel:
            # Per-channel quantization
            out_channels = self.weight.shape[0]
            self.register_buffer("w_scale", torch.ones(out_channels, 1, 1, 1))
            self.register_buffer("w_zp", torch.zeros(out_channels, 1, 1, 1))
        else:
            # Per-tensor quantization
            self.register_buffer("w_scale", torch.tensor(1.0))
            self.register_buffer("w_zp", torch.tensor(0.0))

        self.frozen = False
        self.qmin = None
        self.qmax = None

    @torch.no_grad()
    def freeze(self):
        """Compute quantization parameters from trained weights."""
        w = self.weight.detach()

        if self.per_channel:
            # Per-channel quantization
            w_reshaped = w.reshape(w.shape[0], -1)
            w_min = w_reshaped.min(dim=1).values
            w_max = w_reshaped.max(dim=1).values

            scales, zps = [], []
            for ch_min, ch_max in zip(w_min, w_max):
                scale, zp, qmin, qmax = qparams_from_minmax(
                    ch_min, ch_max, n_bits=self.weight_bits,
                    unsigned=False, symmetric=self.symmetric
                )
                scales.append(scale)
                zps.append(zp)

            self.w_scale = torch.stack(scales).view(-1, 1, 1, 1)
            self.w_zp = torch.stack(zps).view(-1, 1, 1, 1)
        else:
            # Per-tensor quantization
            w_min, w_max = w.min(), w.max()
            scale, zp, qmin, qmax = qparams_from_minmax(
                w_min, w_max, n_bits=self.weight_bits,
                unsigned=False, symmetric=self.symmetric
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
        if self.per_channel:
            # Expand scale and zp to match weight dimensions
            scale_expanded = self.w_scale.expand_as(self.weight)
            zp_expanded = self.w_zp.expand_as(self.weight)

            q = quantize(self.weight, scale_expanded, zp_expanded, self.qmin, self.qmax)
            w_dq = dequantize(q, scale_expanded, zp_expanded)
        else:
            q = quantize(self.weight, self.w_scale, self.w_zp, self.qmin, self.qmax)
            w_dq = dequantize(q, self.w_scale, self.w_zp)

        return F.conv2d(x, w_dq, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)

class AdvancedQuantLinear(nn.Linear):
    """
    Enhanced Linear layer with advanced weight quantization.
    """
    def __init__(self, *args, weight_bits=8, per_channel=False, symmetric=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.per_channel = per_channel
        self.symmetric = symmetric

        if per_channel:
            out_features = self.weight.shape[0]
            self.register_buffer("w_scale", torch.ones(out_features, 1))
            self.register_buffer("w_zp", torch.zeros(out_features, 1))
        else:
            self.register_buffer("w_scale", torch.tensor(1.0))
            self.register_buffer("w_zp", torch.tensor(0.0))

        self.frozen = False
        self.qmin = None
        self.qmax = None

    @torch.no_grad()
    def freeze(self):
        """Compute quantization parameters from trained weights."""
        w = self.weight.detach()

        if self.per_channel:
            w_reshaped = w.reshape(w.shape[0], -1)
            w_min = w_reshaped.min(dim=1).values
            w_max = w_reshaped.max(dim=1).values

            scales, zps = [], []
            for ch_min, ch_max in zip(w_min, w_max):
                scale, zp, qmin, qmax = qparams_from_minmax(
                    ch_min, ch_max, n_bits=self.weight_bits,
                    unsigned=False, symmetric=self.symmetric
                )
                scales.append(scale)
                zps.append(zp)

            self.w_scale = torch.stack(scales).view(-1, 1)
            self.w_zp = torch.stack(zps).view(-1, 1)
        else:
            w_min, w_max = w.min(), w.max()
            scale, zp, qmin, qmax = qparams_from_minmax(
                w_min, w_max, n_bits=self.weight_bits,
                unsigned=False, symmetric=self.symmetric
            )
            self.w_scale.copy_(scale)
            self.w_zp.copy_(zp)

        self.qmin, self.qmax = qmin, qmax
        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            return F.linear(x, self.weight, self.bias)

        if self.per_channel:
            scale_expanded = self.w_scale.expand_as(self.weight)
            zp_expanded = self.w_zp.expand_as(self.weight)

            q = quantize(self.weight, scale_expanded, zp_expanded, self.qmin, self.qmax)
            w_dq = dequantize(q, scale_expanded, zp_expanded)
        else:
            q = quantize(self.weight, self.w_scale, self.w_zp, self.qmin, self.qmax)
            w_dq = dequantize(q, self.w_scale, self.w_zp)

        return F.linear(x, w_dq, self.bias)

# ---------- Enhanced Model Surgery ----------
def advanced_swap_to_quant_modules(model, weight_bits=8, act_bits=8,
                                  activations_unsigned=True, per_channel_weights=False,
                                  weight_symmetric=True, activation_calibration='minmax'):
    """
    Enhanced model surgery with advanced quantization options.
    """
    for name, m in list(model.named_children()):
        # Recursively process child modules
        advanced_swap_to_quant_modules(m, weight_bits, act_bits, activations_unsigned,
                                     per_channel_weights, weight_symmetric, activation_calibration)

        # Replace Conv2d with AdvancedQuantConv2d
        if isinstance(m, nn.Conv2d):
            q = AdvancedQuantConv2d(
                m.in_channels, m.out_channels, m.kernel_size,
                stride=m.stride, padding=m.padding, dilation=m.dilation,
                groups=m.groups, bias=(m.bias is not None),
                weight_bits=weight_bits, per_channel=per_channel_weights,
                symmetric=weight_symmetric
            )
            q.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                q.bias.data.copy_(m.bias.data)
            setattr(model, name, q)

        # Replace Linear with AdvancedQuantLinear
        elif isinstance(m, nn.Linear):
            q = AdvancedQuantLinear(
                m.in_features, m.out_features,
                bias=(m.bias is not None),
                weight_bits=weight_bits, per_channel=per_channel_weights,
                symmetric=weight_symmetric
            )
            q.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                q.bias.data.copy_(m.bias.data)
            setattr(model, name, q)

        # Replace ReLU with ReLU + AdvancedActFakeQuant
        elif isinstance(m, (nn.ReLU, nn.ReLU6)):
            is_relu6 = isinstance(m, nn.ReLU6)
            inplace = getattr(m, "inplace", False)

            if is_relu6:
                relu_layer = nn.ReLU6(inplace=inplace)
            else:
                relu_layer = nn.ReLU(inplace=inplace)

            seq = nn.Sequential(OrderedDict([
                ("relu", relu_layer),
                ("aq", AdvancedActFakeQuant(n_bits=act_bits, unsigned=activations_unsigned,
                                          calibration=activation_calibration)),
            ]))
            setattr(model, name, seq)

def freeze_all_quant(model):
    """
    Freeze all quantization parameters after calibration.
    """
    for mod in model.modules():
        if isinstance(mod, (AdvancedQuantConv2d, AdvancedQuantLinear, AdvancedActFakeQuant)):
            mod.freeze()

# ---------- Enhanced Size Calculation ----------
def model_size_bytes_fp32(model):
    """Calculate total model size if all parameters are FP32."""
    total = 0
    for p in model.parameters():
        total += p.numel() * 4  # 4 bytes per FP32
    return total

def model_size_bytes_quant(model, weight_bits=8):
    """
    Enhanced model size calculation with quantization.
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
    Calculate overhead from quantization metadata.
    """
    metadata_bytes = 0
    for mod in model.modules():
        if isinstance(mod, (AdvancedQuantConv2d, AdvancedQuantLinear)):
            if mod.per_channel:
                # Per-channel: scale and zp for each output channel
                metadata_bytes += mod.w_scale.numel() * 4 * 2  # 4 bytes each for scale and zp
            else:
                metadata_bytes += 8  # scale (4 bytes) + zero_point (4 bytes)
        if isinstance(mod, AdvancedActFakeQuant):
            metadata_bytes += 8  # scale + zero_point
    return metadata_bytes

def calculate_compression_ratios(model, weight_bits=8, act_bits=8):
    """Calculate comprehensive compression statistics."""
    fp32_size = model_size_bytes_fp32(model)
    quant_size = model_size_bytes_quant(model, weight_bits)
    metadata_size = calculate_metadata_size(model)
    total_size = quant_size + metadata_size

    # Weight compression
    total_weights = sum(p.numel() for name, p in model.named_parameters() if "weight" in name)
    weight_fp32_size = total_weights * 4
    weight_quant_size = total_weights * weight_bits // 8
    weight_compression = weight_fp32_size / max(weight_quant_size, 1)

    # Activation compression (estimated)
    # Assuming average activation tensor size and typical network behavior
    activation_params = total_weights * 2  # Rough estimate
    activation_fp32_size = activation_params * 4
    activation_quant_size = activation_params * act_bits // 8
    activation_compression = activation_fp32_size / max(activation_quant_size, 1)

    # Overall compression
    overall_compression = fp32_size / max(total_size, 1)

    return {
        'fp32_size_mb': fp32_size/1024/1024,
        'quant_size_mb': total_size/1024/1024,
        'weight_compression': weight_compression,
        'activation_compression': activation_compression,
        'overall_compression': overall_compression,
        'metadata_kb': metadata_size/1024,
        'weight_size_mb': weight_quant_size/1024/1024,
        'model_size_reduction': 100*(1 - total_size/fp32_size)
    }

def print_compression(model, weight_bits=8, act_bits=8):
    """Print detailed compression statistics."""
    stats = calculate_compression_ratios(model, weight_bits, act_bits)

    print("\n" + "="*70)
    print("ADVANCED COMPRESSION SUMMARY")
    print("="*70)
    print(f"FP32 Model Size:           {stats['fp32_size_mb']:.2f} MB")
    print(f"Quantized Weights:         {stats['weight_size_mb']:.2f} MB ({weight_bits}-bit)")
    print(f"Metadata Overhead:         {stats['metadata_kb']:.2f} KB")
    print(f"Total Quantized Size:      {stats['quant_size_mb']:.2f} MB")
    print(f"-"*70)
    print(f"Weight Compression Ratio:  {stats['weight_compression']:.2f}x")
    print(f"Activation Compression:    {stats['activation_compression']:.2f}x")
    print(f"Overall Compression Ratio: {stats['overall_compression']:.2f}x")
    print(f"Model Size Reduction:      {stats['model_size_reduction']:.1f}%")
    print(f"-"*70)
    print(f"Weight Quantization:       {weight_bits} bits")
    print(f"Activation Quantization:   {act_bits} bits")
    print("="*70 + "\n")

    return stats
