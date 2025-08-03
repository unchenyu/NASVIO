import random
import numpy as np


class ArchEncoder:
    def __init__(
        self,
        ks1_list=[3, 5, 7],
        ks_list=[3, 5],
        expand_list=[i for i in range(1, 9)],
        depth_list=[0, 1],
        ks_stage=3,
        expand_stage=8,
        depth_stage=3,
    ):
        self.ks1_list = ks1_list
        self.ks_list = ks_list
        self.expand_list = expand_list
        self.depth_list = depth_list
        
        self.ks_stage = ks_stage
        self.expand_stage = expand_stage
        self.depth_stage = depth_stage

        # build info dict
        self.n_dim = 0

        self.k_info = dict(val2id=[])
        self.e_info = dict(val2id=[])
        self.d_info = dict(val2id=[])
        self._build_info_dict(target="k")
        self._build_info_dict(target="e")
        self._build_info_dict(target="d")

    def _build_info_dict(self, target):
        if target == "k":
            target_dict = self.k_info
            # for ks1 list
            target_dict["val2id"].append({})
            for k in self.ks1_list:
                target_dict["val2id"][0][k] = self.n_dim
                self.n_dim += 1
            # for rest ks list
            choices = self.ks_list
            for i in range(1, self.ks_stage):
                target_dict["val2id"].append({})
                for k in choices:
                    target_dict["val2id"][i][k] = self.n_dim
                    self.n_dim += 1
        
        elif target == "e":
            target_dict = self.e_info
            choices = self.expand_list
            for i in range(self.expand_stage):
                target_dict["val2id"].append({})
                for k in choices:
                    target_dict["val2id"][i][k] = self.n_dim
                    self.n_dim += 1

        elif target == "d":
            target_dict = self.d_info
            choices = self.depth_list
            for i in range(self.depth_stage):
                target_dict["val2id"].append({})
                for k in choices:
                    target_dict["val2id"][i][k] = self.n_dim
                    self.n_dim += 1

    def arch2feature(self, arch_dict):
        ks, e, d = (
            arch_dict["ks"],
            arch_dict["e"],
            arch_dict["d"],
        )

        feature = np.zeros(self.n_dim)
        for i in range(self.ks_stage):
            feature[self.k_info["val2id"][i][ks[i]]] = 1
        for i in range(self.expand_stage):
            feature[self.e_info["val2id"][i][e[i]]] = 1
        for i in range(self.depth_stage):
            feature[self.d_info["val2id"][i][d[i]]] = 1
        return feature

    def random_sample_arch(self):
        return {
            # "ks": random.choices(self.ks_list, k=self.ks_stage),
            "ks": [random.choice(self.ks1_list)] + random.choices(self.ks_list, k=self.ks_stage-1),
            "e": random.choices(self.expand_list, k=self.expand_stage),
            "d": random.choices(self.depth_list, k=self.depth_stage),
        }

    def mutate_arch(self, arch_dict, mutate_prob):
        for i in range(self.ks_stage):
            if random.random() < mutate_prob:
                if i == 0:
                    arch_dict["ks"][i] = random.choice(self.ks1_list)
                else:
                    arch_dict["ks"][i] = random.choice(self.ks_list)

        for i in range(self.expand_stage):
            if random.random() < mutate_prob:
                arch_dict["e"][i] = random.choice(self.expand_list)

        for i in range(self.depth_stage):
            if random.random() < mutate_prob:
                arch_dict["d"][i] = random.choice(self.depth_list)

        return arch_dict
