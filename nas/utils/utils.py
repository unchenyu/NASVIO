import os
import numpy as np

import torch
import torch.nn as nn


def adjust_bn_according_to_idx(bn, idx):
    bn.weight.data = torch.index_select(bn.weight.data, 0, idx)
    bn.bias.data = torch.index_select(bn.bias.data, 0, idx)
    if type(bn) in [nn.BatchNorm1d, nn.BatchNorm2d]:
        bn.running_mean.data = torch.index_select(bn.running_mean.data, 0, idx)
        bn.running_var.data = torch.index_select(bn.running_var.data, 0, idx)


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


def min_divisible_value(n1, v1):
    """make sure v1 is divisible by n1, otherwise decrease v1"""
    if v1 >= n1:
        return n1
    while n1 % v1 != 0:
        v1 -= 1
    return v1


def build_activation(act_func, inplace=True):
    if act_func == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_func == "relu6":
        return nn.ReLU6(inplace=inplace)
    elif act_func == "lrelu":
        return nn.LeakyReLU(0.1, inplace=inplace)
    elif act_func == "gelu":
        return nn.GELU()
    elif act_func == "tanh":
        return nn.Tanh()
    elif act_func == "sigmoid":
        return nn.Sigmoid()
    elif act_func is None or act_func == "none":
        return None
    else:
        raise ValueError("do not support: %s" % act_func)


def val2list(val, repeat_time=1):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]


def get_net_device(net):
    return net.parameters().__next__().device


def write_log(logs_path, log_str, prefix="valid", should_print=True, mode="a"):
    if not os.path.exists(logs_path):
        os.makedirs(logs_path, exist_ok=True)
    """ prefix: valid, train, test """
    if prefix in ["valid", "test"]:
        with open(os.path.join(logs_path, "valid_console.txt"), mode) as fout:
            fout.write(log_str + "\n")
            fout.flush()
    if prefix in ["valid", "test", "train"]:
        with open(os.path.join(logs_path, "train_console.txt"), mode) as fout:
            if prefix in ["valid", "test"]:
                fout.write("=" * 10)
            fout.write(log_str + "\n")
            fout.flush()
    else:
        with open(os.path.join(logs_path, "%s.txt" % prefix), mode) as fout:
            fout.write(log_str + "\n")
            fout.flush()
    if should_print:
        print(log_str)


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DistributedTensor(object):
    def __init__(self, name):
        self.name = name
        self.sum = None
        self.count = torch.zeros(1)[0]
        self.synced = False

    def update(self, val, delta_n=1):
        val *= delta_n
        if self.sum is None:
            self.sum = val.detach()
        else:
            self.sum += val.detach()
        self.count += delta_n