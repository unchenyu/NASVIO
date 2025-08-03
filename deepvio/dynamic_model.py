import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, trunc_normal_, constant_

from params import par
from deepvio.model import InertialEncoder, PoseRNN

from nas.layers.dynamic_layers import DynamicConvLayer
from nas.layers.layers import ConvLayer
from nas.utils.utils import val2list, adjust_bn_according_to_idx


class VisualEncoderV3(nn.Module):
    def __init__(self, par, convs, visual_head=None, preserve_weight=False):
        super(VisualEncoderV3, self).__init__()
        # CNN
        self.par = par
        self.conv1 = convs[0]
        self.conv2 = convs[1]
        self.conv3 = convs[2]
        if convs[3]:
            self.conv3_1 = convs[3]
        self.conv4 = convs[4]
        if convs[5]:
            self.conv4_1 = convs[5]
        self.conv5 = convs[6]
        if convs[7]:
            self.conv5_1 = convs[7]
        self.conv6 = convs[8]
        self.convs = [conv for conv in convs if conv is not None]
        # Comput the shape based on diff image size
        if visual_head is not None:
            self.visual_head = visual_head
        else:
            __tmp = Variable(torch.zeros(1, 6, par.img_w, par.img_h))
            __tmp = self.encode_image(__tmp)
            self.visual_head = nn.Linear(int(np.prod(__tmp.size())), par.visual_f_len)

        if not preserve_weight:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            kaiming_normal_(m.weight, 0.1)
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            constant_(m.weight, 1)
            constant_(m.bias, 0)

    def forward(self, x, batch_size, seq_len):
        x = self.encode_image(x)
        x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, fv)
        v_f = self.visual_head(x)  # (batch, seq_len, 256)
        return v_f

    def encode_image(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class DynamicVisualEncoderV3(VisualEncoderV3):
    def __init__(self, par, ks_list1=[3,5,7], ks_list=[3,5], 
                 expand_ratio_list=[i for i in range(1,9)], depth_list=[0, 1],
                 base_conv=[8, 16, 32, 32, 64, 64, 64, 64]):
        # CNN
        self.use_norm = par.batch_norm
        if self.use_norm:
            self.bias = False
        else:
            self.bias = True
        norm_func = 'BN'
        act_func = 'lrelu'
        self.ks_list1 = ks_list1
        self.ks_list = ks_list
        self.expand_ratio_list = expand_ratio_list
        self.depth_list = depth_list
        self.base_conv = base_conv

        in_channel = 6
        self.in_channel = in_channel
        out_channel_list = [base_conv[0] * expand for expand in expand_ratio_list]
        conv1 = DynamicConvLayer(
            in_channel_list=[in_channel],
            out_channel_list=out_channel_list,
            kernel_size_list=ks_list1,
            stride=2,
            use_norm=self.use_norm,
            norm_func=norm_func,
            act_func=act_func,
            bias=self.bias,
            )
        in_channel = max(out_channel_list)
        out_channel_list = [base_conv[1] * expand for expand in expand_ratio_list]
        conv2 = DynamicConvLayer(
            in_channel_list=[in_channel],
            out_channel_list=out_channel_list,
            kernel_size_list=ks_list,
            stride=2,
            use_norm=self.use_norm,
            norm_func=norm_func,
            act_func=act_func,
            bias=self.bias,
            )
        in_channel = max(out_channel_list)
        out_channel_list = [base_conv[2] * expand for expand in expand_ratio_list]
        conv3 = DynamicConvLayer(
            in_channel_list=[in_channel],
            out_channel_list=out_channel_list,
            kernel_size_list=ks_list,
            stride=2,
            use_norm=self.use_norm,
            norm_func=norm_func,
            act_func=act_func,
            bias=self.bias,
            )
        in_channel = max(out_channel_list)
        out_channel_list = [base_conv[3] * expand for expand in expand_ratio_list]
        conv3_1 = DynamicConvLayer(
            in_channel_list=[in_channel],
            out_channel_list=out_channel_list,
            kernel_size_list=3,
            stride=1,
            use_norm=self.use_norm,
            norm_func=norm_func,
            act_func=act_func,
            bias=self.bias,
            )
        in_channel = max(out_channel_list)
        out_channel_list = [base_conv[4] * expand for expand in expand_ratio_list]
        conv4 = DynamicConvLayer(
            in_channel_list=[in_channel],
            out_channel_list=out_channel_list,
            kernel_size_list=3,
            stride=2,
            use_norm=self.use_norm,
            norm_func=norm_func,
            act_func=act_func,
            bias=self.bias,
            )
        in_channel = max(out_channel_list)
        out_channel_list = [base_conv[5] * expand for expand in expand_ratio_list]
        conv4_1 = DynamicConvLayer(
            in_channel_list=[in_channel],
            out_channel_list=out_channel_list,
            kernel_size_list=3,
            stride=1,
            use_norm=self.use_norm,
            norm_func=norm_func,
            act_func=act_func,
            bias=self.bias,
            )
        in_channel = max(out_channel_list)
        out_channel_list = [base_conv[6] * expand for expand in expand_ratio_list]
        conv5 = DynamicConvLayer(
            in_channel_list=[in_channel],
            out_channel_list=out_channel_list,
            kernel_size_list=3,
            stride=2,
            use_norm=self.use_norm,
            norm_func=norm_func,
            act_func=act_func,
            bias=self.bias,
            )
        in_channel = max(out_channel_list)
        out_channel_list = [base_conv[7] * expand for expand in expand_ratio_list]
        conv5_1 = DynamicConvLayer(
            in_channel_list=[in_channel],
            out_channel_list=out_channel_list,
            kernel_size_list=3,
            stride=1,
            use_norm=self.use_norm,
            norm_func=norm_func,
            act_func=act_func,
            bias=self.bias,
            )
        in_channel = max(out_channel_list)
        out_channel = 1024
        conv6 = DynamicConvLayer(
            in_channel_list=[in_channel],
            out_channel_list=[out_channel],
            kernel_size_list=3,
            stride=2,
            use_norm=self.use_norm,
            norm_func=norm_func,
            act_func=act_func,
            bias=self.bias,
            )

        self.len_ks = 2
        self.len_e = len(base_conv)
        self.len_d = 3
        # runtime depth
        self.runtime_depth = [max(self.depth_list)] * self.len_d

        super(DynamicVisualEncoderV3, self).__init__(par, [conv1, conv2, conv3, conv3_1, 
                conv4, conv4_1, conv5, conv5_1, conv6])
    
    def encode_image(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        if self.runtime_depth[0]:
            x = self.conv3_1(x)
        x = self.conv4(x)
        if self.runtime_depth[1]:
            x = self.conv4_1(x)
        x = self.conv5(x)
        if self.runtime_depth[2]:
            x = self.conv5_1(x)
        x = self.conv6(x)
        return x

    @ property
    def config(self):
        blocks_config = []
        for block in self.convs:
            blocks_config += block.config
        return {
            "name": DynamicVisualEncoderV3.__name__,
            "blocks": blocks_config,
        }

    """ set, sample and get active sub-networks"""

    def set_max_net(self):
        self.set_active_subnet(ks1=max(self.ks_list1), ks=max(self.ks_list),
                e=max(self.expand_ratio_list), d=max(self.depth_list))

    def set_active_subnet(self, ks1=None, ks=None, e=None, d=None):
        ks = val2list(ks, self.len_ks)
        e = val2list(e, self.len_e)
        d = val2list(d, self.len_d)

        if ks1 is not None:
            self.convs[0].active_kernel_size = ks1
        for i, k in enumerate(ks):
            if k is not None:
                self.convs[i+1].active_kernel_size = k
        for i, expand in enumerate(e):
            if expand is not None:
                self.convs[i].active_out_channel = expand * self.base_conv[i]
        for i, depth in enumerate(d):
            if depth is not None:
                self.runtime_depth[i] = depth

    def sample_active_subnet(self):
        ks1_settings = random.choice(self.ks_list1)
        ks_settings = random.choices(self.ks_list, k=self.len_ks)
        expand_setting = random.choices(self.expand_ratio_list, k=self.len_e)
        depth_setting = random.choices(self.depth_list, k=self.len_d)
        
        self.set_active_subnet(ks1_settings, ks_settings, expand_setting, depth_setting)

        return {
                "ks": [ks1_settings] + ks_settings,
                "e": expand_setting,
                "d": depth_setting,
        }

    def get_active_subnet(self, input_channel=None, preserve_weight=True):
        if input_channel is None:
            input_channel = self.in_channel
        blocks = []
        for i, block in enumerate(self.convs):
            if i in [3, 5, 7] and self.runtime_depth[i//2-1] == 0:
                blocks.append(None)
            else:
                blocks.append(block.get_active_subnet(input_channel, preserve_weight))
                input_channel = block.out_channels

        _subnet = VisualEncoderV3(self.par, blocks, self.visual_head, preserve_weight)
        return _subnet

    def get_active_subnet_config(self, input_channel=None):
        if input_channel is None:
            input_channel = self.in_channel
        block_config_list = []
        for i, block in enumerate(self.convs):
            if i in [3, 5, 7] and self.runtime_depth[i//2-1] == 0:
                continue
            else:
                block_config_list.append(block.get_active_subnet_config(input_channel))
            input_channel = block.out_channels

        return {
            "name": VisualEncoderV3.__name__,
            "blocks": block_config_list,
        }


class DeepVIO(nn.Module):
    def __init__(self, par, visual_encoder, inertial_encoder=None, pose_net=None, preserve_weight=False):
        super().__init__()

        self.par = par
        self.visual_encoder = visual_encoder
        self.inertial_encoder = InertialEncoder(par) if inertial_encoder is None else inertial_encoder
        self.pose_net = PoseRNN(par) if pose_net is None else pose_net

        if not preserve_weight:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    kaiming_normal_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(0)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param.data[start:end].fill_(1.)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, t_x, t_i, prev=None):
        v_f, imu = self.encoder_forward(t_x, t_i)
        batch_size = v_f.shape[0]
        seq_len = v_f.shape[1]

        angle_list, trans_list = [], []
        hidden = torch.zeros(batch_size, par.rnn_hidden_size).to(t_x.device) if prev is None else prev[0].contiguous()[:, -1, :]

        for i in range(seq_len):
            angle, trans, prev, _ = self.pose_net(v_f[:, i, :].unsqueeze(1), imu[:, i, :].unsqueeze(1), prev)
            angle_list.append(angle)
            trans_list.append(trans)
            hidden = prev[0].contiguous()[:, -1, :]
        angles = torch.cat(angle_list, dim=1)
        trans = torch.cat(trans_list, dim=1)

        return angles, trans, prev

    def encoder_forward(self, v, imu):
        # x: (batch, seq_len, channel, width, height)
        # stack_image

        v = torch.cat((v[:, :-1], v[:, 1:]), dim=2)
        batch_size = v.size(0)
        seq_len = v.size(1)

        # CNN
        v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4))
        v_f = self.visual_encoder(v, batch_size, seq_len)

        ll = 11 + self.par.imu_prev
        imu = torch.cat([imu[:, i * 10:i * 10 + ll, :].unsqueeze(1) for i in range(seq_len)], dim=1)
        imu = self.inertial_encoder(imu)

        return v_f, imu


class DynamicDeepVIOV3(DeepVIO):
    def __init__(self, par):
        visual_encoder = DynamicVisualEncoderV3(par, ks_list1=par.ks_list1,
                    ks_list=par.ks_list, expand_ratio_list=par.expand_ratio_list,
                    depth_list=par.depth_list)
        super(DynamicDeepVIOV3, self).__init__(par, visual_encoder)

    @ property
    def config(self):
        blocks_config = self.visual_encoder.config["blocks"]
        return {
            "name": DynamicDeepVIO.__name__,
            "blocks": blocks_config,
        }

    """ set, sample and get active sub-networks"""

    def set_max_net(self):
        self.visual_encoder.set_max_net()

    def set_active_subnet(self, ks1=None, ks=None, e=None, d=None):
        self.visual_encoder.set_active_subnet(ks1, ks, e, d)

    def sample_active_subnet(self):
        return self.visual_encoder.sample_active_subnet()

    def get_active_subnet(self, input_channel=None, preserve_weight=True):
        visual_encoder = self.visual_encoder.get_active_subnet(preserve_weight=preserve_weight)

        _subnet = DeepVIO(self.par, visual_encoder, self.inertial_encoder, self.pose_net, preserve_weight)
        return _subnet

    def get_active_subnet_config(self, input_channel=None):
        block_config_list = self.visual_encoder.get_active_subnet_config()["blocks"]

        return {
            "name": DeepVIO.__name__,
            "blocks": block_config_list,
        }


if __name__ == "__main__":
    from params import par
    net = DynamicVisualEncoderV3(par)
    print(net)
    dummy_input = torch.rand(1, 6, 512, 256)
    output = net.encode_image(dummy_input)
    print(output.shape)
    print(net.sample_active_subnet())
    print(net.get_active_subnet_config())
    subnet = net.get_active_subnet()
    print(subnet)
    output = net.encode_image(dummy_input)
    print(output.shape)