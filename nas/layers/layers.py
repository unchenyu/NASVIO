import torch
import torch.nn as nn

from collections import OrderedDict
from nas.utils.utils import build_activation, get_same_padding, min_divisible_value


def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvLayer.__name__: ConvLayer,
        LinearLayer.__name__: LinearLayer,
    }

    layer_name = layer_config.pop("name")
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        use_norm=True,
        norm_func='LN',
        norm_chan_per_group=None,
        use_act=True,
        act_func="relu",
        dropout_rate=0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_norm = use_norm
        self.norm_func = norm_func
        self.use_act = use_act
        self.act_func = act_func
        self.dropout_rate = dropout_rate

        # default normal 3x3_Conv with bn and relu
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        """ modules """
        modules = {}
        # norm layer
        if self.use_norm:
            if norm_func == "BN":
                modules["norm"] = nn.BatchNorm2d(out_channels)
            if norm_func == "LN":
                modules["norm"] = nn.GroupNorm(1, out_channels)
            if norm_func == "GN":
                modules["norm"] = nn.GroupNorm(out_channels//norm_chan_per_group, out_channels)
        else:
            modules["norm"] = None
        # activation
        if use_act:
            modules["act"] = build_activation(
                self.act_func, self.use_norm
            )
        else:
            modules["act"] = None
        # dropout
        if self.dropout_rate > 0:
            modules["dropout"] = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            modules["dropout"] = None
        # weight
        modules["weight"] = self.weight_op()

        # add modules
        for op in ["weight", "norm", "act"]:
            if modules[op] is None:
                continue
            elif op == "weight":
                # dropout before weight operation
                if modules["dropout"] is not None:
                    self.add_module("dropout", modules["dropout"])
                for key in modules["weight"]:
                    self.add_module(key, modules["weight"][key])
            else:
                self.add_module(op, modules[op])

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict(
            {
                "conv": nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=padding,
                    dilation=self.dilation,
                    groups=min_divisible_value(self.in_channels, self.groups),
                    bias=self.bias,
                )
            }
        )
        return weight_dict

    def forward(self, x):
        # similar to nn.Sequential
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            if self.dilation > 1:
                conv_str = "%dx%d_DilatedConv" % (kernel_size[0], kernel_size[1])
            else:
                conv_str = "%dx%d_Conv" % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                conv_str = "%dx%d_DilatedGroupConv" % (kernel_size[0], kernel_size[1])
            else:
                conv_str = "%dx%d_GroupConv" % (kernel_size[0], kernel_size[1])
        conv_str += "_O%d" % self.out_channels
        conv_str += "_" + self.act_func.upper()
        if self.use_norm:
            if isinstance(self.norm, nn.GroupNorm):
                conv_str += "_GN%d" % self.norm.num_groups
            elif isinstance(self.norm, nn.BatchNorm2d):
                conv_str += "_BN"
        return conv_str

    @property
    def config(self):
        return {
            "name": ConvLayer.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "groups": self.groups,
            "bias": self.bias,
            "use_norm": self.use_norm,
            "norm_func": self.norm_func,
            "norm_chan_per_group": self.norm_chan_per_group,
            "use_act": self.use_act,
            "act_func": self.act_func,
            "dropout_rate": self.dropout_rate,
        }

    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)