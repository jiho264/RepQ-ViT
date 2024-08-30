import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def lp_loss(pred, tgt, p=2.0, reduction="none"):
    """
    loss function measured in L_p Norm
    """
    if reduction == "none":
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


class UniformQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """

    def __init__(self, n_bits: int = 8, channel_wise: bool = False):
        super(UniformQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, "bitwidth not supported"
        self.n_bits = n_bits
        self.n_levels = 2**self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise

    def __repr__(self):
        s = super(UniformQuantizer, self).__repr__()
        s = (
            "("
            + s
            + " inited={}, channel_wise={})".format(self.inited, self.channel_wise)
        )
        return s

    def forward(self, x: torch.Tensor):

        if self.inited is False:
            self.delta, self.zero_point = self.init_quantization_scale(
                x, self.channel_wise
            )
            self.inited = True

        # start quantization
        x_int = torch.round(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=-1)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(
                        x_clone[:, :, c], channel_wise=False
                    )
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(
                        x_clone[c], channel_wise=False
                    )
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
            else:
                raise NotImplementedError
        else:
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()

            # delta = (x_max - x_min) / (2**self.n_bits - 1)
            # zero_point = (-x_min / delta).round()
            best_score = 1e10
            for pct in [0.999, 0.9999, 0.99999]:
                try:
                    new_max = torch.quantile(x_clone.reshape(-1), pct)
                    new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                except:
                    new_max = torch.tensor(
                        np.percentile(x_clone.reshape(-1).cpu(), pct * 100),
                        device=x_clone.device,
                        dtype=torch.float32,
                    )
                    new_min = torch.tensor(
                        np.percentile(x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                        device=x_clone.device,
                        dtype=torch.float32,
                    )
                x_q = self.quantize(x_clone, new_max, new_min)
                score = lp_loss(x_clone, x_q, p=2, reduction="all")
                if score < best_score:
                    best_score = score
                    delta = (new_max - new_min) / (2**self.n_bits - 1)
                    zero_point = (-new_min / delta).round()

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2**self.n_bits - 1)
        zero_point = (-min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q


class LogSqrt2Quantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """

    def __init__(self, n_bits: int = 8, channel_wise: bool = False):
        super(LogSqrt2Quantizer, self).__init__()
        assert 2 <= n_bits <= 8, "bitwidth not supported"
        self.n_bits = n_bits
        self.n_levels = 2**self.n_bits
        self.delta = None
        self.inited = False
        self.channel_wise = channel_wise
        self.k = None
        self.b = None

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            # self.delta = self.init_quantization_scale(x)
            self.init_quantization_scale(x)
            self.inited = True

        # start quantization
        # x_dequant = self.quantize(x, self.delta)
        x_dequant = self.myquantize(x, None, self.k, self.b)
        return x_dequant

    def myquantize(self, x, delta, k, b, verbose=False):
        delta = torch.tensor(2).pow(31)
        x_int = (x * delta).round()
        x_int_max = x_int.max()

        if b == 0:
            x_int_log = -1 * (x_int.log2().round() - x_int_max.log2().round() + k)
        else:
            x_int_log = (
                -1
                * (
                    (x_int.log2().round() - x_int_max.log2().round() + k) / b.log2()
                ).round()
            )

        if verbose:
            print("[1] log2\n", x_int_log.unique(), torch.unique(x_int_log).numel())

        ## Big INT --------------------------------------------- Small INT
        # [1] log2
        # tensor([-3., -2., -1., -0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.,
        #         11., 12., 13., inf], device='cuda:0') 18
        x_int_log = x_int_log.clamp(0, self.n_levels - 1)

        if verbose:
            print("[2] clamped\n", x_int_log.unique(), torch.unique(x_int_log).numel())

        # x_power_2 = (2 ** (-x_int_log) * (self.n_levels - 1)) / self.n_levels / k
        # x_power_2 = 2 ** (-x_int_log)
        x_power_2 = b ** (-x_int_log)
        # s_x = 1 / (self.n_levels) / k
        if verbose:
            print("[3] encoded\n", x_power_2.unique(), torch.unique(x_power_2).numel())
            print()

        return x_power_2

    def init_quantization_scale(self, x: torch.Tensor):
        x_clone = x.clone().detach()
        best_score = 1e10
        best_k = 0
        best_b = 0
        for k in range(-10, 10):
            for b in range(2, 15):
                b = torch.tensor(b)
                x_q = self.myquantize(x_clone, None, k, b)
                score = lp_loss(x_clone, x_q, p=2, reduction="all")
                if score < best_score:
                    best_score = score
                    best_k = k
                    best_b = b

        _ = self.myquantize(x_clone, None, best_k, best_b, verbose=True)
        self.k = best_k
        self.b = best_b
        print(x.shape)
        print(f" best score: {best_score}, k: {best_k}, log base: {best_b}")

    # """below is org code"""
    # def init_quantization_scale(self, x: torch.Tensor):
    #     delta = None
    #     x_clone = x.clone().detach()
    #     delta = x_clone.max()
    #     best_score = 1e10
    #     for pct in [0.999, 0.9999, 0.99999]:  #
    #         try:
    #             new_delta = torch.quantile(x_clone.reshape(-1), pct)
    #         except:
    #             new_delta = torch.tensor(
    #                 np.percentile(x_clone.reshape(-1).cpu(), pct * 100),
    #                 device=x_clone.device,
    #                 dtype=torch.float32,
    #             )
    #         x_q = self.quantize(x_clone, new_delta)
    #         score = lp_loss(x_clone, x_q, p=2, reduction="all")
    #         if score < best_score:
    #             best_score = score
    #             delta = new_delta

    #     return delta

    # def quantize(self, x, delta):
    #     from math import sqrt

    #     x_int = torch.round(-1 * (x / delta).log2() * 2)
    #     mask = x_int >= self.n_levels
    #     x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
    #     # print(x_quant.unique(), x_quant.unique().numel())
    #     odd_mask = (x_quant % 2) * (sqrt(2) - 1) + 1
    #     x_float_q = 2 ** (-1 * torch.ceil(x_quant / 2)) * odd_mask * delta
    #     x_float_q[mask] = 0
    #     # print(x_float_q.unique(), x_float_q.unique().numel())
    #     return x_float_q
