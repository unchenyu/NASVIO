import os
import copy
import torch


def count_conv_flop(out_h, out_w, in_channels, out_channels, kernel_size, groups=1):
    return in_channels * out_channels * kernel_size * kernel_size * out_h * out_w / groups


def count_fc_flop(in_features, out_features):
    return in_features * out_features


class FLOPsModel:
    def __init__(self, size=[256, 512], base_conv=[8, 16, 32, 32, 64, 64, 64, 64]):
        self.size = size
        self.base_conv = base_conv

    def get_efficiency(self, arch_dict):
        ks = arch_dict["ks"]
        e = arch_dict["e"]
        d = arch_dict["d"] if arch_dict["d"] is not None else [1, 1, 1]
        flops = 0
        h = self.size[0]
        w = self.size[1]
        # conv1
        flops += count_conv_flop(h//2, w//2, 6, e[0]*self.base_conv[0], ks[0])
        h, w = h//2, w//2
        # conv2
        flops += count_conv_flop(h//2, w//2, e[0]*self.base_conv[0], e[1]*self.base_conv[1], ks[1])
        h, w = h//2, w//2
        # conv3
        flops += count_conv_flop(h//2, w//2, e[1]*self.base_conv[1], e[2]*self.base_conv[2], ks[2])
        h, w = h//2, w//2
        # conv3_1
        if d[0]:
            flops += count_conv_flop(h, w, e[2]*self.base_conv[2], e[3]*self.base_conv[3], 3)
        # conv4
        flops += count_conv_flop(h//2, w//2, e[3]*self.base_conv[3], e[4]*self.base_conv[4], 3)
        h, w = h//2, w//2
        # conv4_1
        if d[1]:
            flops += count_conv_flop(h, w, e[4]*self.base_conv[4], e[5]*self.base_conv[5], 3)
        # conv5
        flops += count_conv_flop(h//2, w//2, e[5]*self.base_conv[5], e[6]*self.base_conv[6], 3)
        h, w = h//2, w//2
        # conv5_1
        if d[2]:
            flops += count_conv_flop(h, w, e[6]*self.base_conv[6], e[7]*self.base_conv[7], 3)
        # conv6
        flops += count_conv_flop(h//2, w//2, e[7]*self.base_conv[7], 1024, 3)
        
        return flops / 1e9 # GFLOPs 


if __name__ == '__main__':
    efficiency_predictor = FLOPsModel()
    arch_dict = {
        # 'ks': [7, 5, 5],
        # 'e': [8, 8, 8, 8, 8, 8, 8, 8],
        # 'd': None
        # 'ks': [3, 3, 3],
        # 'e': [1, 1, 1, 1, 1, 1, 1, 1],
        # 'd': None
        # 'd': [0, 0, 0]
        'ks': [7, 3, 5],
        'e': [1, 1, 1, 2, 1, 2, 4, 1],
        'd': [1, 1, 1],
    }
    print(efficiency_predictor.get_efficiency(arch_dict))

    # Original model FLOPs is 7.747G, minimum model FLOPs is 0.109G