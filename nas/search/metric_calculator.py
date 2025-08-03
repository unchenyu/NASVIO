import os
import copy
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from params import par
from deepvio.data_helper import get_data_info, ImageSequenceDataset, ImageSequenceDataset_test
from nas.utils.bn_utils import set_running_statistics

device = par.devices[0]


class MetricCalculator:
    def __init__(self, dynamic_net):
        self.dynamic_net = dynamic_net.to(device)
        self.test_dl, self.subset_dl = self.dataset()


    def dataset(self):
        test_len = 8 * par.seq_len[0]
        test_batch = 4

        # Generate subset of the training set for setting batchnorm statistics
        train_df = get_data_info(par.train_video, par.seq_len[0], overlap=par.overlap, shuffle=True, sort=True, is_right=False)
        train_dataset = ImageSequenceDataset(train_df)
        indices = random.choices(range(len(train_dataset)), k=200)
        subset = torch.utils.data.Subset(train_dataset, indices)
        subset_dl = DataLoader(subset, batch_size=test_batch, shuffle=False, num_workers=par.n_processors, pin_memory=par.pin_mem, drop_last=True)

        # Prepare the data
        test_df = get_data_info(par.valid_video, test_len, overlap=1, shuffle=False, sort=True, is_right=False)

        test_sampler = None
        test_dataset = ImageSequenceDataset_test(test_df)
        test_dl = DataLoader(test_dataset, batch_sampler=test_sampler, batch_size=test_batch, shuffle=False, num_workers=par.n_processors, pin_memory=par.pin_mem, drop_last=True)

        return test_dl, subset_dl

    def validate(self):
        # st_t = time.time()
        self.dynamic_net.eval()

        loss_ang_mean_test, loss_trans_mean_test, loss_mean_test = 0, 0, 0
    
        with torch.no_grad():
            for _, v_x, _, v_i, v_y in self.test_dl:

                v_x = v_x.to(device)
                v_y = v_y.to(device)
                v_i = v_i.to(device)

                angle, trans, _ = self.dynamic_net(v_x, v_i, prev=None)
                angle_loss = torch.nn.functional.mse_loss(angle, v_y[:, :, :3])
                translation_loss = torch.nn.functional.mse_loss(trans, v_y[:, :, 3:])
                # loss = 100 * angle_loss + translation_loss
                loss = 50 * angle_loss + translation_loss

                loss = loss.data.cpu().numpy()
                loss_mean_test += float(loss)
                loss_ang_mean_test += float(angle_loss)
                loss_trans_mean_test += float(translation_loss)

        # print('Valid take {:.1f} sec'.format(time.time() - st_t))
        loss_ang_mean_test /= len(self.test_dl)
        loss_trans_mean_test /= len(self.test_dl)
        loss_mean_test /= len(self.test_dl)

        # print(f'Test loss mean: {loss_mean_test:.7f}, test ang loss mean: {loss_ang_mean_test*100:.7f}, test trans loss mean: {loss_trans_mean_test:.7f}')
        return loss_ang_mean_test, loss_trans_mean_test, loss_mean_test

    def calculate_metric(self, arch_dict_list):
        metric = []
        for arch_dict in arch_dict_list:
            arch_dict = copy.deepcopy(arch_dict)
            self.dynamic_net.set_active_subnet(ks1=arch_dict['ks'][0], ks=arch_dict['ks'][1:], e=arch_dict['e'], d=arch_dict['d'])
            set_running_statistics(self.dynamic_net, self.subset_dl)
            ang_mean, trans_mean, loss_mean = self.validate()
            metric.append([ang_mean, trans_mean, loss_mean])

        return metric

    def predict_metric(self, arch_dict_list):
        return self.calculate_metric(arch_dict_list)


if __name__ == "__main__":
    from deepvio.dynamic_model import DynamicDeepVIO, DynamicDeepVIOV3
    dynamic_net = DynamicDeepVIOV3(par)
    net_checkpoint_path = './models/bn/depth/t00010204060809_v050710_im256x512_s11x11_b16_decay5e-06_C+I_vf_512_if_256_0_conv_cat_flip.model.epoch_241'
    init = torch.load(net_checkpoint_path, map_location="cpu")
    dynamic_net.load_state_dict(init)

    metric_calculator = MetricCalculator(dynamic_net)
    arch_dict = [{
        'ks': [7, 5, 5],
        'e': [8, 8, 8, 8, 8, 8, 8, 8],
        'd': [1, 1, 1]
        # 'ks': [3, 3, 3],
        # 'e': [1, 1, 1, 1, 1, 1, 1, 1],
        # 'd': [0, 0, 0]
    }]
    print(metric_calculator.calculate_metric(arch_dict))