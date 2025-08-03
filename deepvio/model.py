import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_

from params import par


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )


# The inertial encoder for raw imu data
class InertialEncoder(nn.Module):
    def __init__(self, par):
        super(InertialEncoder, self).__init__()
        self.method = par.imu_method
        if self.method == 'bi-LSTM':
            self.rnn_imu_head = nn.Linear(6, 128)
            self.encoder = nn.LSTM(
                input_size=128,
                hidden_size=par.imu_hidden_size,
                num_layers=2,
                dropout=par.rnn_dropout_between,
                batch_first=True,
                bidirectional=True)
        elif self.method == 'conv':
            self.encoder_conv = nn.Sequential(
                nn.Conv1d(6, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(par.dropout),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(par.dropout),
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(par.dropout))
            len_f = par.imu_per_image + 1 + par.imu_prev
            #len_f = (len_f - 1) // 2 // 2 + 1
            self.proj = nn.Linear(256 * 1 * len_f, par.imu_f_len)

    def forward(self, x):
        # x: (N, seq_len, 11, 6)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))    # x: (N x seq_len, 11, 6)
        if self.method == 'bi-LSTM':
            x = self.rnn_imu_head(x)    # x: (N x seq_len, 11, 128)
            x, hc = self.encoder(x)     # x: (N x seq_len, 11, 2, 128)
            x = x.view(x.shape[0], x.shape[1], 2, -1)
            out = torch.cat((x[:, 0, 0, :], x[:, -1, 1, :]), -1)   # out: (N x seq_len, 256)
            return out.view(batch_size, seq_len, 256)
        elif self.method == 'conv':
            x = self.encoder_conv(x.permute(0, 2, 1))    # x: (N x seq_len, 64, 11)
            out = self.proj(x.view(x.shape[0], -1))      # out: (N x seq_len, 256)
            return out.view(batch_size, seq_len, 256)


# The fusion module
class FusionModule(nn.Module):
    def __init__(self, par, temp=None):
        super(FusionModule, self).__init__()
        self.fuse_method = par.fuse_method
        self.f_len = par.imu_f_len + par.visual_f_len
        if self.fuse_method == 'soft':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, self.f_len))
        elif self.fuse_method == 'hard':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, 2 * self.f_len))
            if temp is None:
                self.temp = 1
            else:
                self.temp = temp

    def forward(self, v, i):
        if self.fuse_method == 'cat':
            return torch.cat((v, i), -1)
        elif self.fuse_method == 'soft':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            return feat_cat * weights
        elif self.fuse_method == 'hard':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            weights = weights.view(v.shape[0], v.shape[1], self.f_len, 2)
            mask = F.gumbel_softmax(weights, tau=1, hard=True, dim=-1)
            return feat_cat * mask[:, :, :, 0]


# The pose estimation network
class PoseRNN(nn.Module):
    def __init__(self, par):
        super(PoseRNN, self).__init__()

        # The main RNN network
        f_len = par.visual_f_len + par.imu_f_len
        self.rnn = nn.LSTM(
            input_size=f_len,
            hidden_size=par.rnn_hidden_size,
            num_layers=2,
            dropout=par.rnn_dropout_between,
            batch_first=True)

        self.fuse = FusionModule(par)

        # The output networks
        self.rnn_drop_out = nn.Dropout(par.rnn_dropout_out)
        # self.regressor = nn.Sequential(
        #    nn.Linear(par.rnn_hidden_size, 6))
        self.regressor = nn.Sequential(
            nn.Linear(par.rnn_hidden_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6))

    def forward(self, visual_f, imu_f, prev=None):

        if prev is not None:
            prev = (prev[0].transpose(1, 0).contiguous(), prev[1].transpose(1, 0).contiguous())

        batch_size = visual_f.shape[0]
        seq_len = visual_f.shape[1]
        
        fused = self.fuse(visual_f, imu_f)
        #self.rnn.flatten_parameters()
        out, hc = self.rnn(fused) if prev is None else self.rnn(fused, prev)
        out = self.rnn_drop_out(out)
        pose = self.regressor(out)
        angle = pose[:, :, :3]
        trans = pose[:, :, 3:]

        hc = (hc[0].transpose(1, 0).contiguous(), hc[1].transpose(1, 0).contiguous())
        return angle, trans, hc, out


class Encoder(nn.Module):
    def __init__(self, par):
        super(Encoder, self).__init__()
        # CNN
        self.par = par
        self.batchNorm = par.batch_norm
        self.conv1 = conv(self.batchNorm, 6, 64, kernel_size=7, stride=2, dropout=par.conv_dropout[0])
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2, dropout=par.conv_dropout[1])
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2, dropout=par.conv_dropout[2])
        self.conv3_1 = conv(self.batchNorm, 256, 256, kernel_size=3, stride=1, dropout=par.conv_dropout[3])
        self.conv4 = conv(self.batchNorm, 256, 512, kernel_size=3, stride=2, dropout=par.conv_dropout[4])
        self.conv4_1 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1, dropout=par.conv_dropout[5])
        self.conv5 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=2, dropout=par.conv_dropout[6])
        self.conv5_1 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1, dropout=par.conv_dropout[7])
        self.conv6 = conv(self.batchNorm, 512, 1024, kernel_size=3, stride=2, dropout=par.conv_dropout[8])
        # Comput the shape based on diff image size
        __tmp = Variable(torch.zeros(1, 6, par.img_w, par.img_h))
        __tmp = self.encode_image(__tmp)

        self.visual_head = nn.Linear(int(np.prod(__tmp.size())), par.visual_f_len)
        self.inertial_encoder = Inertial_encoder(par)

    def forward(self, v, imu):
        # x: (batch, seq_len, channel, width, height)
        # stack_image

        v = torch.cat((v[:, :-1], v[:, 1:]), dim=2)
        batch_size = v.size(0)
        seq_len = v.size(1)

        # CNN
        v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4))
        v_high = self.encode_image(v)
        v_high = v_high.view(batch_size, seq_len, -1)  # (batch, seq_len, fv)
        v_high = self.visual_head(v_high)  # (batch, seq_len, 256)

        ll = 11 + self.par.imu_prev
        imu = torch.cat([imu[:, i * 10:i * 10 + ll, :].unsqueeze(1) for i in range(seq_len)], dim=1)
        imu = self.inertial_encoder(imu)

        return v_high, imu

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6


class DeepVIO(nn.Module):
    def __init__(self, par):
        super(DeepVIO, self).__init__()

        self.feature_net = Encoder(par)
        self.pose_net = PoseRNN(par)

    def forward(self, t_x, t_i, prev=None):
        v_f, imu = self.feature_net(t_x, t_i)
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