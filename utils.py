import numpy as np
from sklearn import metrics
from collections import namedtuple
import os
import time
import random
import torch
import pickle

Data = namedtuple('data', ['samples', 'nodes'])


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def graph_rmse(target, graph, avg=10):
    base_samples = target.observation_sample(num=avg)

    target_samples = target.pass_forward(base_samples)
    graph_samples = graph.pass_forward(base_samples)

    rmse_loss = np.sqrt(metrics.mean_squared_error(target_samples, graph_samples))
    return rmse_loss


class MetricCollector():
    def __init__(self, mmetrics):
        self.metric_dict = {m: [] for m in mmetrics}

    def add(self, m, value):
        self.metric_dict[m].append(value)

    def add_all(self, d):
        for k, v in d.items():
            self.add(k, v)

    def output_avg(self):
        output = {k: [np.mean(v), np.var(v)] for k, v in self.metric_dict.items()}
        return output


class MetaMetricCollector():
    def __init__(self, mmetrics):
        self.metric_mean_dict = {m: [] for m in mmetrics}
        self.metric_var_dict = {m: [] for m in mmetrics}

    def add_performance(self, performance):
        for k, (mean, var) in performance.items():
            self.metric_mean_dict[k].append(mean)
            self.metric_var_dict[k].append(var)

    def output_avg(self):
        mean_output = {k: [np.mean(v), np.var(v)] for k, v in self.metric_mean_dict.items()}
        var_output = {k: [np.mean(v), np.var(v)] for k, v in self.metric_var_dict.items()}

        return mean_output, var_output


def find_best_fidelity(capacity, fidelity_cost_list):
    for i in range(len(fidelity_cost_list)):
        if capacity < fidelity_cost_list[i + 1]:
            return i


def check_dir_and_make(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_local_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def write_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
