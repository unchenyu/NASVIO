import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from nas.utils.utils import get_same_padding, sub_filter_start_end


class DynamicConv2d(nn.Module):
    KERNEL_TRANSFORM_MODE = 1

    def __init__(
        self, max_in_channels, max_out_channels, kernel_size_list, stride=1, dilation=1, bias=False,
    ):
        super(DynamicConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.dilation = dilation
        self.bias = bias

        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_out_channels,
            max(self.kernel_size_list),
            stride=self.stride,
            bias=bias,
        )

        if len(self.kernel_size_list) > 1:
            self._ks_set = list(set(self.kernel_size_list))
            self._ks_set.sort()  # e.g., [3, 5, 7]
            if self.KERNEL_TRANSFORM_MODE is not None:
                # register scaling parameters
                # 7to5_matrix, 5to3_matrix
                scale_params = {}
                for i in range(len(self._ks_set) - 1):
                    ks_small = self._ks_set[i]
                    ks_larger = self._ks_set[i + 1]
                    param_name = "%dto%d" % (ks_larger, ks_small)
                    # noinspection PyArgumentList
                    scale_params["%s_matrix" % param_name] = Parameter(
                        torch.eye(ks_small ** 2)
                    )
                for name, param in scale_params.items():
                    self.register_parameter(name, param)

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_out_channel = self.max_out_channels

    def get_active_filter(self, out_channel, in_channel, kernel_size=None):
        if isinstance(in_channel, int):
            in_channel = slice(0, in_channel)
        if kernel_size is None:
            return self.conv.weight[:out_channel, in_channel, :, :]

        max_kernel_size = max(self.kernel_size_list)

        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv.weight[:out_channel, in_channel, start:end, start:end]
        if self.KERNEL_TRANSFORM_MODE is not None and kernel_size < max_kernel_size:
            start_filter = self.conv.weight[
                :out_channel, in_channel, :, :
            ]  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(
                    _input_filter.size(0), _input_filter.size(1), -1
                )
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter,
                    self.__getattr__("%dto%d_matrix" % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks ** 2
                )
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks, target_ks
                )
                start_filter = _input_filter
            filters = start_filter
        return filters

    def get_active_bias(self, out_channel):
        return self.conv.bias[:out_channel] if self.bias else None

    def forward(self, x, out_channel=None, kernel_size=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        if kernel_size is None:
            kernel_size = self.active_kernel_size

        in_channel = x.size(1)
        if len(self.kernel_size_list) > 1:
            filters = self.get_active_filter(out_channel, in_channel, kernel_size).contiguous()
        else:
            filters = self.get_active_filter(out_channel, in_channel).contiguous()

        bias = self.get_active_bias(out_channel)

        padding = get_same_padding(kernel_size)
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)
        return y


class DynamicBatchNorm2d(nn.Module):
    SET_RUNNING_STATISTICS = False

    def __init__(self, max_feature_dim):
        super(DynamicBatchNorm2d, self).__init__()

        self.max_feature_dim = max_feature_dim
        self.norm = nn.BatchNorm2d(self.max_feature_dim)

    @staticmethod
    def bn_forward(x, bn: nn.BatchNorm2d, feature_dim):
        if bn.num_features == feature_dim or DynamicBatchNorm2d.SET_RUNNING_STATISTICS:
            return bn(x)
        else:
            exponential_average_factor = 0.0

            if bn.training and bn.track_running_stats:
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                    if bn.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = bn.momentum
            return F.batch_norm(
                x,
                bn.running_mean[:feature_dim],
                bn.running_var[:feature_dim],
                bn.weight[:feature_dim],
                bn.bias[:feature_dim],
                bn.training or not bn.track_running_stats,
                exponential_average_factor,
                bn.eps,
            )

    def forward(self, x):
        feature_dim = x.size(1)
        y = self.bn_forward(x, self.norm, feature_dim)
        return y


class DynamicGroupNorm(nn.GroupNorm):
    def __init__(
        self, num_groups, num_channels, eps=1e-6, affine=True, channel_per_group=None
    ):
        super(DynamicGroupNorm, self).__init__(num_groups, num_channels, eps, affine)
        self.channel_per_group = channel_per_group
        self.num_groups = num_groups

    def forward(self, x):
        n_channels = x.size(1)
        if self.channel_per_group is not None:
            n_groups = n_channels // self.channel_per_group
        else:
            n_groups = self.num_groups
        return F.group_norm(
            x, n_groups, self.weight[:n_channels], self.bias[:n_channels], self.eps
        )

    @property
    def norm(self):
        return self