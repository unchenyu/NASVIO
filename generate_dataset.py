import os
import argparse
import random
import time
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from params import par
from deepvio.dynamic_model import DynamicDeepVIOV3
from deepvio.data_helper import get_data_info, ImageSequenceDataset, ImageSequenceDataset_test
from nas.utils.bn_utils import set_running_statistics


device = par.devices[0]


def dataset(par):
    test_len = 8 * par.seq_len[0]
    test_batch = 4

    print('Create new data info')
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

    print('=' * 50)

    return test_dl, subset_dl


def validate(dynamic_net, test_dl):
    # st_t = time.time()
    dynamic_net.eval()

    loss_ang_mean_test, loss_trans_mean_test, loss_mean_test = 0, 0, 0
    
    with torch.no_grad():
        for _, v_x, _, v_i, v_y in test_dl:

            v_x = v_x.to(device)
            v_y = v_y.to(device)
            v_i = v_i.to(device)

            angle, trans, _ = dynamic_net(v_x, v_i, prev=None)
            angle_loss = torch.nn.functional.mse_loss(angle, v_y[:, :, :3])
            translation_loss = torch.nn.functional.mse_loss(trans, v_y[:, :, 3:])
            # loss = 100 * angle_loss + translation_loss
            loss = 50 * angle_loss + translation_loss

            loss = loss.data.cpu().numpy()
            loss_mean_test += float(loss)
            loss_ang_mean_test += float(angle_loss)
            loss_trans_mean_test += float(translation_loss)

    # print('Valid take {:.1f} sec'.format(time.time() - st_t))
    loss_ang_mean_test /= len(test_dl)
    loss_trans_mean_test /= len(test_dl)
    loss_mean_test /= len(test_dl)

    print(f'Test loss mean: {loss_mean_test:.7f}, test ang loss mean: {loss_ang_mean_test*100:.7f}, test trans loss mean: {loss_trans_mean_test:.7f}')
    return loss_ang_mean_test, loss_trans_mean_test, loss_mean_test


def load_models(dynamic_net, model_path=None):
    init = torch.load(model_path, map_location="cpu")
    dynamic_net.load_state_dict(init)
    print('Checkpoint loaded')


def net_setting2id(net_setting):
    return json.dumps(net_setting)


def net_id2setting(net_id):
    return json.loads(net_id)


def net_id_path(path):
    return os.path.join(path, "net_id.dict")


def metric_dict_path(path):
    return os.path.join(path, "metric.dict")


def main(args, dynamic_net, test_dl, subset_dl):
    # Randomly generate subnets
    if os.path.isfile(net_id_path(args.path)):
        net_id_list = json.load(open(net_id_path(args.path)))
    else:
        net_id_list = set()
        seed = args.manual_seed
        while len(net_id_list) < args.n_arch:
            net_setting = dynamic_net.sample_active_subnet()
            net_id = net_setting2id(net_setting)
            net_id_list.add(net_id)
            seed += 1
        net_id_list = list(net_id_list)
        net_id_list.sort()
        json.dump(net_id_list, open(net_id_path(args.path), "w"), indent=4)

    # Inference all random subnets
    metric_save_path = metric_dict_path(args.path)
    metric_dict = {}
    if os.path.isfile(metric_dict_path(args.path)):
        existing_metric_dict = json.load(open(metric_dict_path(args.path)))
    else:
        existing_metric_dict = {}
    for net_id in net_id_list:
        net_setting = net_id2setting(net_id)
        key = net_setting2id({**net_setting})
        if key in existing_metric_dict:
            metric_dict[key] = existing_metric_dict[key]
            continue
        # dynamic_net.set_active_subnet(**net_setting)
        dynamic_net.set_active_subnet(ks1=net_setting['ks'][0], ks=net_setting['ks'][1:], e=net_setting['e'], d=net_setting['d'])
        set_running_statistics(dynamic_net, subset_dl)
        angle_loss, trans_loss, total_loss = validate(dynamic_net, test_dl)
        metric_dict.update({key: {'angle_loss':angle_loss, 'trans_loss':trans_loss, 'total_loss':total_loss}})
        json.dump(metric_dict, open(metric_save_path, "w"), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.path = './dataset/depth'
    os.makedirs(args.path, exist_ok=True)

    # Dynamic model
    dynamic_net = DynamicDeepVIOV3(par)

    # args.net_checkpoint_path = './models/bn/expand/t00010204060809_v050710_im256x512_s11x11_b16_decay5e-06_C+I_vf_512_if_256_0_conv_cat_flip.model.epoch_254'
    args.net_checkpoint_path = './models/bn/depth/t00010204060809_v050710_im256x512_s11x11_b16_decay5e-06_C+I_vf_512_if_256_0_conv_cat_flip.model.epoch_241'

    args.manual_seed = 0
    args.n_arch = 4000

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    load_models(dynamic_net, model_path=args.net_checkpoint_path)
    dynamic_net = dynamic_net.to(device)

    test_dl, subset_dl = dataset(par)

    main(args, dynamic_net, test_dl, subset_dl)