import math

import torch
import torch.nn.functional as F
from torch import nn

# SIREN network implementation
# Paper: https://arxiv.org/abs/2006.09661
# Implementation: https://github.com/lucidrains/siren-pytorch


# sin activation
class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


# siren layer
class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0=30., c=6., is_first=False, use_bias=True, activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


class SirenBaseNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, decoder_layers=0, decoder_dim=1024, w0=30., w0_initial=30., use_bias=True, final_activation=None):
        super().__init__()
        layers = []
        decoders = []
        self.dim_in = dim_in
        self.dim_out = dim_out * 8
        self.dim_feat = dim_out
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layers.append(Siren(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                use_bias=use_bias,
                is_first=is_first
            ))

        if decoder_layers:
            for ind in range(decoder_layers):
                is_first = ind == 0
                layer_w0 = w0_initial if is_first else w0
                layer_dim_in = dim_in if is_first else decoder_dim

                decoders.append(Siren(
                    dim_in=layer_dim_in,
                    dim_out=decoder_dim,
                    w0=layer_w0,
                    use_bias=use_bias,
                    is_first=is_first
                ))
        else:
            decoders.append(nn.Identity())

        self.net = nn.Sequential(*layers)
        self.occ_net = nn.Sequential(*decoders)
        self.last_layer_lat = Siren(dim_in=dim_hidden, dim_out=self.dim_out, w0=w0, use_bias=use_bias, activation=final_activation)
        self.last_layer_occ = Siren(dim_in=decoder_dim, dim_out=2, w0=w0, use_bias=use_bias, activation=final_activation)

    def forward(self, x):
        x = self.net(x)
        lat = self.last_layer_lat(x).view(x.shape[0], -1, self.dim_feat)
        occ = self.last_layer_occ(self.occ_net(lat))

        return lat, occ


class SirenDecoder(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=30., w0_initial=30., use_bias=True, final_activation=None):
        super().__init__()
        layers = []
        self.dim_in = dim_in
        self.dim_out = dim_out
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layers.append(Siren(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                use_bias=use_bias,
                is_first=is_first
            ))

        self.net = nn.Sequential(*layers)
        self.last_layer = Siren(dim_in=dim_hidden, dim_out=self.dim_out, w0=w0, use_bias=use_bias, activation=final_activation)

    def forward(self, x):
        x = self.net(x)
        x = self.last_layer(x)
        return x
