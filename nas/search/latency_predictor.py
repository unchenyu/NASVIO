import os
import copy
import torch
import time

from fvcore.nn import FlopCountAnalysis

from params import par
from deepvio.dynamic_model import DynamicVisualEncoderV3, DynamicDeepVIOV3


class LatencyModel:
    def __init__(self, dynamic_net=None, h=256, w=512, device="cuda:0"):
        self.dynamic_net = DynamicVisualEncoderV3(par) if dynamic_net is None else dynamic_net
        self.device = device
        # Default inputs
        self.inputs = []
        for _ in range(10):
            # self.inputs.append(torch.randn((1, 6, h, w)).to(self.device))
            self.inputs.append(torch.randn((1, 6, h, w)))

    def validate(self, subnet):
        with torch.no_grad():
            for img in self.inputs:
                x = subnet.encode_image(img)

    def get_efficiency(self, arch_dict):
        ks = arch_dict["ks"]
        e = arch_dict["e"]
        d = arch_dict["d"]
        self.dynamic_net.set_active_subnet(ks1=ks[0], ks=ks[1:], e=e, d=d)
        # subnet = self.dynamic_net.get_active_subnet().to(self.device)
        subnet = self.dynamic_net.get_active_subnet()
        subnet.eval()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self.validate(subnet)
        end.record()
        torch.cuda.synchronize()
        time = start.elapsed_time(end)
        del subnet

        return time #GFLOPs


if __name__ == '__main__':
    # efficiency_predictor = LatencyModel(DynamicVisualEncoderV3(par))
    efficiency_predictor = LatencyModelV2(DynamicDeepVIOV3(par))
    arch_dict = {
        # 'ks': [7, 5, 5],
        # 'e': [8, 8, 8, 8, 8, 8, 8, 8],
        # 'd': [1, 1, 1]
        # 'ks': [3, 3, 3],
        # 'e': [1, 1, 1, 1, 1, 1, 1, 1],
        # 'd': [1, 1, 1]
        # # mminimum FLOPs
        # 'ks': [5, 5, 3],
        # 'e': [1, 1, 1, 2, 1, 1, 1, 1],
        # 'd': [1, 1, 1]
        # mminimum latency
        'ks': [5, 3, 5],
        'e': [1, 1, 2, 2, 1, 1, 3, 1],
        'd': [0, 1, 1]
    }
    print(efficiency_predictor.get_efficiency(arch_dict))

    # Original model FLOPs is 7.747G, minimum model FLOPs is 0.109G