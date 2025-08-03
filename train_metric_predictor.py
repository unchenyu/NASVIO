import os
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from nas.search.metric_predictor.metric_dataset import MetricDataset
from nas.search.metric_predictor.metric_predictor import MetricPredictor
from nas.search.arch_encoder import ArchEncoder
from nas.utils.utils import AverageMeter, write_log


def save_model(save_path, checkpoint=None, is_best=False, model_name=None):
    if model_name is None:
        model_name = "checkpoint.pth.tar"

    latest_fname = os.path.join(save_path, "latest.txt")
    model_path = os.path.join(save_path, model_name)
    with open(latest_fname, "w") as fout:
        fout.write(model_path + "\n")
    torch.save(checkpoint, model_path)

    if is_best:
        best_path = os.path.join(save_path, "model_best.pth.tar")
        torch.save({"state_dict": checkpoint["state_dict"], "base_metric": checkpoint["base_metric"]}, best_path)


def train(net, train_loader, valid_loader, args):
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()
    # criterion = nn.MSELoss()
    # criterion = nn.L1Loss()

    val_criterion = nn.L1Loss()

    best_criterion = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(net, train_loader, epoch, criterion, optimizer, args)

        if (epoch + 1) % args.validation_frequency == 0:
            val_loss = validate(net, valid_loader, epoch, val_criterion, args)

            is_best = val_loss < best_criterion
            best_criterion = min(best_criterion, val_loss)

            val_log = "Valid [{0}/{1}]\tloss {2:.6f}\t{3:.6f}".format(
                    epoch + 1,
                    args.epochs,
                    np.mean(val_loss),
                    best_criterion,
                )
            val_log += "\tTrain loss {train_loss:.6f}\t".format(train_loss=train_loss)
            write_log(args.logs_path, val_log, prefix="valid", should_print=False)
        else:        
            is_best = False

        save_model(
            save_path = args.ckpt_path,
            checkpoint = {
                    "epoch": epoch,
                    "best_metric": best_criterion,
                    "optimizer": optimizer.state_dict(),
                    "state_dict": net.state_dict(),
                    "base_metric": net.base_metric.item(),
                },
                is_best=is_best,
            )


def train_one_epoch(net, train_loader, epoch, criterion, optimizer, args):
    net.train()
    nBatch = len(train_loader)
    losses = AverageMeter()

    with tqdm(
            total=nBatch,
            desc="Train Epoch #{}".format(epoch + 1),
        ) as t:
        for i, (features, targets) in enumerate(train_loader):
            features, targets = features.to(args.device), targets.to(args.device)
            output = net(features)
            loss = criterion(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), features.size(0))

            t.set_postfix(
                {
                    "loss": losses.avg,
                }
            )
            t.update(1)
    return losses.avg


def validate(net, valid_loader, epoch, test_criterion, args):
    net.eval()

    losses = AverageMeter()

    with torch.no_grad():
        with tqdm(
                total=len(valid_loader),
                desc="Validate Epoch #{}".format(epoch + 1),
            ) as t:
            for i, (features, targets) in enumerate(valid_loader):
                features, targets = features.to(args.device), targets.to(args.device)
                # ompute output
                output = net(features)
                loss = test_criterion(output, targets)

                losses.update(loss.item(), features.size(0))

                t.set_postfix(
                    {
                        "loss": losses.avg,
                    }
                )
                t.update(1)
    return losses.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.path = './dataset/depth'
    args.ckpt_path = './dataset/pred_checkpoint'
    args.logs_path = './dataset/logs'
    os.makedirs(args.path, exist_ok=True)
    os.makedirs(args.ckpt_path, exist_ok=True)
    os.makedirs(args.logs_path, exist_ok=True)

    args.ks1_list = "3,5,7"
    args.ks_list = "3,5"
    args.expand_list = "1,2,3,4,5,6,7,8"
    args.depth_list = "0,1"

    args.epochs = 500
    args.validation_frequency = 10
    args.lr = 1e-3
    args.device = "cuda:0"

    args.manual_seed = 0

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    args.ks1_list = [int(ks) for ks in args.ks1_list.split(",")]
    args.ks_list = [int(ks) for ks in args.ks_list.split(",")]
    args.expand_list = [int(e) for e in args.expand_list.split(",")]
    args.depth_list = [int(d) for d in args.depth_list.split(",")]

    metric_dataset = MetricDataset(args.path)
    arch_manager = ArchEncoder(args.ks1_list, args.ks_list, args.expand_list, args.depth_list)
    train_loader, valid_loader, base_metric = metric_dataset.build_metric_data_loader(arch_manager, n_training_sample=3800)
    metric_predictor = MetricPredictor(arch_manager, base_metric=base_metric)

    train(metric_predictor, train_loader, valid_loader, args)