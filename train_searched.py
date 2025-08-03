import os
import time
import math
import random
import glob
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from params import par
from deepvio.dynamic_model import DynamicDeepVIOV3
from deepvio.data_helper import get_data_info, get_data_info_test, ImageSequenceDataset, ImageSequenceDataset_test
from nas.utils.bn_utils import set_running_statistics

device = par.devices[0]


def dataset(par):
    test_len = 8 * par.seq_len[0]
    test_batch = 4

    print('Create new data info')
    # Prepare the data
    partition = par.partition
    train_df = get_data_info(par.train_video, par.seq_len[0], overlap=par.overlap, shuffle=True, sort=True, is_right=False)
    test_df = get_data_info(par.valid_video, test_len, overlap=1, shuffle=False, sort=True, is_right=False)

    #train_sampler = SortedRandomBatchSampler(train_df, par.batch_size, drop_last=True)
    train_sampler = None
    train_dataset = ImageSequenceDataset(train_df)
    train_dl = DataLoader(train_dataset, batch_sampler=train_sampler, batch_size=par.batch_size, shuffle=True, num_workers=par.n_processors, pin_memory=par.pin_mem, drop_last=True)

    # Generate subset of the training set for setting batchnorm statistics
    indices = random.choices(range(len(train_dataset)), k=200)
    subset = torch.utils.data.Subset(train_dataset, indices)
    subset_dl = DataLoader(subset, batch_size=par.batch_size, shuffle=True, num_workers=par.n_processors, pin_memory=par.pin_mem, drop_last=True)

    test_sampler = None
    test_dataset = ImageSequenceDataset_test(test_df)
    test_dl = DataLoader(test_dataset, batch_sampler=test_sampler, batch_size=test_batch, shuffle=False, num_workers=par.n_processors, pin_memory=par.pin_mem, drop_last=True)

    print('Number of samples in training dataset: ', len(train_df.index))
    print('=' * 50)

    return train_dl, test_dl, subset_dl


def train(par, VIONet, train_dl, test_dl):
    # Create optimizer
    optimizer = torch.optim.Adam(VIONet.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=par.decay)
    VIONet = torch.nn.DataParallel(VIONet, par.devices)

    validate(VIONet, test_dl)

    print('Start training: ', par.experiment_name + '  ' + par.name)
    min_loss_t = 1e10
    min_loss_test = 1e10
    min_ang_loss_test = 1e10
    min_trans_loss_test = 1e10

    # epochs_warmup, epochs_joint, epochs_fine = 0, 40, 20
    epochs_warmup, epochs_joint, epochs_fine = 0, 0, 20
    lr_joint, lr_fine = 5e-5, 1e-6

    for ep in range(epochs_warmup + epochs_joint + epochs_fine):

        # Adjust the learning rate
        if ep == epochs_warmup:
            optimizer.param_groups[0]['lr'] = lr_joint
        if ep == epochs_warmup + epochs_joint:
            optimizer.param_groups[0]['lr'] = lr_fine
            
        st_t = time.time()
        print('=' * 50)
        VIONet.train()

        loss_ang_mean_train, loss_trans_mean_train, loss_mean_train = 0, 0, 0
        count_turn, average_train_turn = 0, 0
        count = 0

        for _, t_x, _, t_i, t_y in train_dl:

            t_x = t_x.to(device)
            t_y = t_y.to(device)
            t_i = t_i.to(device)

            optimizer.zero_grad()

            # Forward
            angle, trans, _ = VIONet(t_x, t_i, prev=None)
            angle_loss = torch.nn.functional.mse_loss(angle, t_y[:, :, :3])
            translation_loss = torch.nn.functional.mse_loss(trans, t_y[:, :, 3:])
            # loss = 100 * angle_loss + translation_loss
            loss = 50 * angle_loss + translation_loss
            loss.backward()
            optimizer.step()

            loss = loss.data.cpu().numpy()
            angle_loss = angle_loss.data.cpu().numpy()
            translation_loss = translation_loss.data.cpu().numpy()

            average_train_turn += torch.sum(((angle[:, :, 1] - t_y[:, :, 1])[abs(t_y[:, :, 1]) > 0.03])**2).item()
            count_turn += torch.sum(abs(t_y[:, :, 1]) > 0.03).detach().item()

            loss_ang_mean_train += float(angle_loss)
            loss_trans_mean_train += float(translation_loss)
            loss_mean_train += float(loss)

            count += 1
            if count % 50 == 0:
                print(f'Epoch:{ep}, Iteration: {count}, Loss: {float(loss):.6f}, angle: {float(angle_loss*100):.6f}, trans: {float(translation_loss):.6f}, Train take {time.time()-st_t:.1f} sec')

        loss_ang_mean_train /= len(train_dl)
        loss_trans_mean_train /= len(train_dl)
        loss_mean_train /= len(train_dl)
        average_train_turn /= count_turn

        print(f'Epoch {ep + 1}\ntrain loss mean: {loss_mean_train:.7f}, train ang loss mean: {loss_ang_mean_train*100:.7f}, train trans loss mean: {loss_trans_mean_train:.7f}, err_turn: {average_train_turn*100:.7f}')
        f = open(par.record_path, 'a')
        f.write(f'Epoch {ep + 1}\ntrain loss mean: {loss_mean_train:.7f}, train ang loss mean: {loss_ang_mean_train*100:.7f}, train trans loss mean: {loss_trans_mean_train:.7f}, err_turn: {average_train_turn*100:.7f}\n')

        validate(VIONet, test_dl)
        
        if ep >= epochs_warmup + epochs_joint - 1:
            torch.save(VIONet.module.state_dict(), par.save_model_path + '.epoch_' + str(ep))

        if ep == epochs_warmup - 1 or ep == epochs_warmup + epochs_joint - 1:
            torch.save(optimizer.state_dict(), par.save_optimzer_path + '.epoch_' + str(ep))

        if ep >= epochs_warmup + epochs_joint - 1 and (ep + 1) % 20 == 0:
            torch.save(optimizer.state_dict(), par.save_optimzer_path + '.epoch_' + str(ep))


def validate(VIONet, test_dl):

    st_t = time.time()
    VIONet.eval()

    loss_ang_mean_test, loss_trans_mean_test, loss_ang_mean_raw, loss_mean_test = 0, 0, 0, 0
    count_turn, average_test_turn = 0, 0
    
    with torch.no_grad():
        for _, v_x, _, v_i, v_y in test_dl:

            v_x = v_x.to(device)
            v_y = v_y.to(device)
            v_i = v_i.to(device)

            angle, trans, _ = VIONet(v_x, v_i, prev=None)
            angle_loss = torch.nn.functional.mse_loss(angle, v_y[:, :, :3])
            translation_loss = torch.nn.functional.mse_loss(trans, v_y[:, :, 3:])
            # loss = 100 * angle_loss + translation_loss
            loss = 50 * angle_loss + translation_loss

            loss = loss.data.cpu().numpy()
            loss_ang_mean_test += float(angle_loss)
            loss_trans_mean_test += float(translation_loss)
            loss_ang_mean_raw += float(torch.nn.functional.mse_loss(angle[:, :, 1], v_y[:, :, 1]))
            loss_mean_test += float(loss)

            average_test_turn += torch.sum(((angle[:, :, 1] - v_y[:, :, 1])[abs(v_y[:, :, 1]) > 0.03])**2).item()
            count_turn += torch.sum(abs(v_y[:, :, 1]) > 0.03).detach().item()

    print('Valid take {:.1f} sec'.format(time.time() - st_t))
    loss_ang_mean_test /= len(test_dl)
    loss_trans_mean_test /= len(test_dl)
    loss_ang_mean_raw /= len(test_dl)
    loss_mean_test /= len(test_dl)
    average_test_turn /= count_turn

    print(f'test loss mean: {loss_mean_test:.7f}, test ang loss mean: {loss_ang_mean_test*100:.7f}, test trans loss mean: {loss_trans_mean_test:.7f}, test raw: {loss_ang_mean_raw:.9f}, err_turn: {average_test_turn*100:.7f}')
    f = open(par.record_path, 'a')
    f.write(f'test loss mean: {loss_mean_test:.7f}, test ang loss mean: {loss_ang_mean_test*100:.7f}, test trans loss mean: {loss_trans_mean_test:.7f}, test raw: {loss_ang_mean_raw:.9f}, err_turn: {average_test_turn*100:.7f}\n')


# def map_ckpt(model_dict, ckpt_dict):
#     for key in ckpt_dict:
#         if 'visual_encoder' in key and '.norm.' in key:
#             new_key = key.replace('.norm.norm.', '.norm.')
#             if 'num_batches_tracked' in key:
#                 model_dict[new_key] = ckpt_dict[key]
#             else:
#                 model_dict[new_key] = ckpt_dict[key][:len(model_dict[new_key])]
#     return model_dict

def map_ckpt(model_dict, ckpt_dict):
    for key in ckpt_dict:
        if 'visual_encoder' in key and '.norm.' in key:
            new_key = key.replace('.norm.norm.', '.norm.')
            if new_key in model_dict:
                if 'num_batches_tracked' in key:
                    model_dict[new_key] = ckpt_dict[key]
                else:
                    model_dict[new_key] = ckpt_dict[key][:len(model_dict[new_key])]
    return model_dict


def main(par):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    # Model
    DyVIONet = DynamicDeepVIOV3(par)

    # Load supernet ceckpoint
    # pretrained_w = torch.load('./models/bn/expand/t00010204060809_v050710_im256x512_s11x11_b16_decay5e-06_C+I_vf_512_if_256_0_conv_cat_flip.model.epoch_254', map_location='cpu')
    pretrained_w = torch.load('./models/bn/depth/t00010204060809_v050710_im256x512_s11x11_b16_decay5e-06_C+I_vf_512_if_256_0_conv_cat_flip.model.epoch_241', map_location='cpu')
    DyVIONet.load_state_dict(pretrained_w)

    train_dl, test_dl, subset_dl = dataset(par)

    # # Set minimum net
    # DyVIONet.set_active_subnet(3, 3, 1)

    # # Set maximum net
    # DyVIONet.set_active_subnet(7, 5, 8)

    # Set maximum net for DynamicDeepVIOV3
    DyVIONet.set_max_net()

    # # set searched net for DynamicDeepVIOV3 with metric constraint = 4.2, lowest latency on CPU
    # ks1 = 5
    # ks = [3, 5]
    # e = [1, 1, 2, 2, 1, 1, 3, 1]
    # d = [0, 1, 1]

    # # set searched net for DynamicDeepVIOV3 with metric constraint = 4.2, lowest FLOPs
    # ks1 = 5
    # ks = [5, 3]
    # e = [1, 1, 1, 2, 1, 1, 1, 1]
    # d = [1, 1, 1]


    DyVIONet.set_active_subnet(ks1, ks, e, d)

    DyVIONet.to(device)
    set_running_statistics(DyVIONet, subset_dl)
    validate(DyVIONet, test_dl)

    DyVIONet.cpu()
    VIONet = DyVIONet.get_active_subnet()

    # model_dict = VIONet.state_dict()
    # model_dict = map_ckpt(model_dict, pretrained_w)
    # VIONet.load_state_dict(model_dict)
    # del pretrained_w

    VIONet = VIONet.to(device)

    torch.save(VIONet.state_dict(), par.save_model_path + '.epoch_')

    train(par, VIONet, train_dl, test_dl)


if __name__ == "__main__":
    # main(par)

    from fvcore.nn import FlopCountAnalysis

    DyVIONet = DynamicDeepVIOV3(par)
    # DyVIONet.set_max_net()
    # ks1 = 5
    # ks = [3, 5]
    # e = [1, 1, 2, 2, 1, 1, 3, 1]
    # d = [0, 1, 1]
    # ks1 = 5
    # ks = [5, 3]
    # e = [1, 1, 1, 2, 1, 1, 1, 1]
    # d = [1, 1, 1]
    # DyVIONet.set_active_subnet(ks1, ks, e, d)
    VIONet = DyVIONet.get_active_subnet().cuda()

    image = torch.randn((1, 2, 3, 256, 512)).cuda()
    imu = torch.randn((1, 110, 6)).cuda()
    flops = FlopCountAnalysis(VIONet, (image, imu))
    # flops.tracer_warnings('none')
    # flops.unsupported_ops_warnings(False)
    print(flops.total())