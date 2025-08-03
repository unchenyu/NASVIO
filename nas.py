import os
import argparse
import random
import numpy as np
import warnings

import torch

from params import par
from deepvio.dynamic_model import DynamicDeepVIOV3

from nas.search.metric_predictor.metric_predictor import MetricPredictor
from nas.search.efficiency_predictor import FLOPsModel, FLOPsModelv2
from nas.search.latency_predictor import LatencyModel
from nas.search.metric_calculator import MetricCalculator
from nas.search.arch_encoder import ArchEncoder
from nas.search.evolution import EvolutionFinder, EvolutionFinderV2


def load_models(dynamic_net, model_path=None):
    init = torch.load(model_path, map_location="cpu")
    dynamic_net.load_state_dict(init)


def create_metric_predictor(arch_manager, model_path=None):
    model_dict = torch.load(model_path, map_location="cpu")
    predictor = MetricPredictor(arch_manager, base_metric=model_dict["base_metric"])
    predictor.load_state_dict(model_dict["state_dict"])
    return predictor


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.path = './evolution'
    os.makedirs(args.path, exist_ok=True)

    args.ks1_list = "3,5,7"
    args.ks_list = "3,5"
    args.expand_list = "1,2,3,4,5,6,7,8"
    args.depth_list = "0,1"

    args.manual_seed = 0

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    args.ks1_list = [int(ks) for ks in args.ks1_list.split(",")]
    args.ks_list = [int(ks) for ks in args.ks_list.split(",")]
    args.expand_list = [int(e) for e in args.expand_list.split(",")]
    args.depth_list = [int(d) for d in args.depth_list.split(",")]

    # Dynamic model
    # dynamic_net = DynamicDeepVIOV3(par)
    # args.net_checkpoint_path = './models/bn/depth/t00010204060809_v050710_im256x512_s11x11_b16_decay5e-06_C+I_vf_512_if_256_0_conv_cat_flip.model.epoch_241'

    # load_models(dynamic_net, model_path=args.net_checkpoint_path)

    arch_manager = ArchEncoder(args.ks1_list, args.ks_list, args.expand_list, args.depth_list)
    # efficiency_predictor = FLOPsModel()
    efficiency_predictor = LatencyModel()

    # metric_calculator = MetricCalculator(dynamic_net)

    args.metric_checkpoint_path = './dataset/pred_checkpoint/model_best.pth.tar'
    metric_predictor = create_metric_predictor(arch_manager, args.metric_checkpoint_path)
    metric_predictor.eval()

    evolution = EvolutionFinderV2(arch_manager, efficiency_predictor, metric_predictor, args.path)
    best_valids, best_info = evolution.run_evolution_search(constraint=4.4, verbose=True)

    # arch_dict = [best_info[1]]
    # actual_metric = metric_calculator.calculate_metric(arch_dict)
    # print("Actual metric is: ", actual_metric)