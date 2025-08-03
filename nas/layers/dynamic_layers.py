import torch
import torch.nn as nn
from collections import OrderedDict

from nas.layers.ops import DynamicConv2d, DynamicBatchNorm2d, DynamicGroupNorm
from nas.layers.layers import LinearLayer, ConvLayer, set_layer_from_config
from nas.utils.utils import get_net_device, val2list, build_activation


def copy_bn(target_bn, src_bn):
    feature_dim = (
        target_bn.num_channels
        if isinstance(target_bn, nn.GroupNorm)
        else target_bn.num_features
    )

    target_bn.weight.data.copy_(src_bn.weight.data[:feature_dim])
    target_bn.bias.data.copy_(src_bn.bias.data[:feature_dim])
    if type(src_bn) in [nn.BatchNorm1d, nn.BatchNorm2d]:
        target_bn.running_mean.data.copy_(src_bn.running_mean.data[:feature_dim])
        target_bn.running_var.data.copy_(src_bn.running_var.data[:feature_dim])
        target_bn.num_batches_tracked.data.copy_(src_bn.num_batches_tracked.data)


class DynamicConvLayer(nn.Module):
    def __init__(
        self,
        in_channel_list,
        out_channel_list,
        kernel_size_list=3,
        stride=1,
        dilation=1,
        use_norm=True,
        norm_func="LN",
        use_act=True,
        act_func="relu",
        bias=False,
    ):
        super().__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.kernel_size_list = val2list(kernel_size_list)
        self.stride = stride
        self.dilation = dilation
        self.use_norm = use_norm
        self.norm_func = norm_func
        self.use_act = use_act
        self.act_func = act_func
        self.bias = bias

        self.conv = DynamicConv2d(
            max_in_channels=max(self.in_channel_list),
            max_out_channels=max(self.out_channel_list),
            kernel_size_list=self.kernel_size_list,
            stride=self.stride,
            dilation=self.dilation,
            bias=bias,
        )

        self.norm_chan_per_group = None
        if self.use_norm:
            if norm_func == "BN":
                self.norm = DynamicBatchNorm2d(max(self.out_channel_list))
            if norm_func == "LN":
                self.norm = DynamicGroupNorm(1, max(self.out_channel_list))
            if norm_func == "GN":
                self.norm_chan_per_group = max(self.out_channel_list) // 32
                self.norm = DynamicGroupNorm(32, max(self.out_channel_list),
                                channel_per_group=self.norm_chan_per_group)

        if self.use_act:
            self.act = build_activation(self.act_func)

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        self.conv.active_out_channel = self.active_out_channel
        self.conv.active_kernel_size = self.active_kernel_size

        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        if self.use_act:
            x = self.act(x)
        return x

    @property
    def module_str(self):
        return "DyConv(O%d, K%d, S%d)" % (
            self.active_out_channel,
            self.kernel_size,
            self.stride,
        )

    @property
    def config(self):
        return {
            "name": DynamicConvLayer.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "kernel_size_list": self.kernel_size_list,
            "stride": self.stride,
            "dilation": self.dilation,
            "use_norm": self.use_norm,
            "norm_func": self.norm_func,
            "use_act": self.use_act,
            "act_func": self.act_func,
            "bias": self.bias,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicConvLayer(**config)

    ############################################################################################

    @property
    def in_channels(self):
        return max(self.in_channel_list)

    @property
    def out_channels(self):
        return self.active_out_channel

    @property
    def kernel_size(self):
        return self.active_kernel_size

    ############################################################################################

    # Add support for channel selection on concatenated input [A, B, ...] -> Conv -> C
    # Argument: concatenated_channels, [(start_index, # of channels), (start_index, # of channels), ...]
    def get_active_subnet(self, in_channel, preserve_weight=True, concatenated_channels=None):
        sub_layer = set_layer_from_config(self.get_active_subnet_config(in_channel))
        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        if concatenated_channels is None:
            sub_layer.conv.weight.data.copy_(
                self.conv.get_active_filter(self.active_out_channel, in_channel, self.active_kernel_size).data
            )
        else:
            channel_idx = 0
            for start_idx, n_channels in concatenated_channels:
                sub_layer.conv.weight.data[:, channel_idx:channel_idx+n_channels].copy_(
                    self.conv.get_active_filter(self.active_out_channel,
                    	slice(start_idx, start_idx+n_channels), self.active_kernel_size).data
                    )
                channel_idx += n_channels
        if self.bias:
            sub_layer.conv.bias.data.copy_(
                    self.conv.get_active_bias(self.active_out_channel).data
                )

        if self.use_norm:
            copy_bn(sub_layer.norm, self.norm.norm)

        return sub_layer

    def get_active_subnet_config(self, in_channel):
        return {
            "name": ConvLayer.__name__,
            "in_channels": in_channel,
            "out_channels": self.active_out_channel,
            "kernel_size": self.active_kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "use_norm": self.use_norm,
            "norm_func": self.norm_func,
            "norm_chan_per_group": self.norm_chan_per_group,
            "use_act": self.use_act,
            "act_func": self.act_func,
            "bias": self.bias,
        }