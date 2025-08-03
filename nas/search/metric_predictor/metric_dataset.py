import os
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data


def net_setting2id(net_setting):
    return json.dumps(net_setting)


def net_id2setting(net_id):
    return json.loads(net_id)


class RegDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        super(RegDataset, self).__init__()
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return self.inputs.size(0)


class MetricDataset:
    def __init__(self, path):
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    @property
    def metric_dict_path(self):
        return os.path.join(self.path, "metric.dict")

    def build_metric_data_loader(
        self, arch_encoder, n_training_sample=None, batch_size=64, n_workers=8
    ):
        # load data
        metric_dict = json.load(open(self.metric_dict_path))
        X_all = []
        Y_all = []
        with tqdm(total=len(metric_dict), desc="Loading data") as t:
            for k, v in metric_dict.items():
                dic = json.loads(k)
                X_all.append(arch_encoder.arch2feature(dic))
                Y_all.append(v['total_loss']*10000)
                t.update()
        base_metric = np.mean(Y_all)
        # convert to torch tensor
        X_all = torch.tensor(X_all, dtype=torch.float)
        Y_all = torch.tensor(Y_all, dtype=torch.float)

        # random shuffle
        shuffle_idx = torch.randperm(len(X_all))
        X_all = X_all[shuffle_idx]
        Y_all = Y_all[shuffle_idx]

        # split data
        if n_training_sample is None:
            n_training_sample = X_all.size(0) // 5 * 4
        X_train, Y_train = X_all[:n_training_sample], Y_all[:n_training_sample]
        X_test, Y_test = X_all[n_training_sample:], Y_all[n_training_sample:]
        print("Train Size: %d," % len(X_train), "Valid Size: %d" % len(X_test))

        # build data loader
        train_dataset = RegDataset(X_train, Y_train)
        val_dataset = RegDataset(X_test, Y_test)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=n_workers,
        )
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=n_workers,
        )

        return train_loader, valid_loader, base_metric
