import numpy as np


class ReplayBuffer():
    def __init__(self, config):
        self.config = config
        self.fidelity_num = len(config['fidelity_cost_list'])
        self.sample_list = []  # Store for samples x.
        self.j_list = []
        self.v_list = []
        self.m_list = []

    def total(self):
        return len(self.sample_list)

    def add(self, x, j, v, m):
        self.sample_list.append(x)
        self.j_list.append(j)
        self.v_list.append(v)
        self.m_list.append(m)

    def add_obs(self, x):
        self.add(x, -1, 0.0, self.fidelity_num - 1)

    def add_int(self, x, j, v, m):
        self.add(x, j, v, m)

    def sample_batch(self, num):
        if num <= self.total():
            index_list = np.random.choice(self.total(), size=num, replace=False)
        else:
            index_list = np.arange(self.total())
        samples_batch, j_batch, v_batch, m_batch = [], [], [], []
        for i in index_list:
            samples_batch.append(self.sample_list[i])
            j_batch.append(self.j_list[i])
            v_batch.append(self.v_list[i])
            m_batch.append(self.m_list[i])
        samples_batch = np.array(samples_batch)
        j_batch = np.array(j_batch)
        v_batch = np.array(v_batch)
        m_batch = np.array(m_batch)

        return samples_batch, j_batch, v_batch, m_batch


def get_graph_prior(dataset):
    if dataset == 'ERGraph':
        return 'er'
    if dataset == 'SFGraph':
        return 'sf'
    return 'er'


def get_exp_edges(config):
    if config['dataset'] == 'ERGraph' or config['dataset'] == 'SFGraph':
        return config['dataset_param']['exp_edges']
    else:
        return 2


class Fidelity_Min():
    def __init__(self, config):
        self.config = config
        self.fidelity_cost_list = config['fidelity_cost_list']

    def obtain_fidelity(self):
        return 0


class Fidelity_Max():
    def __init__(self, config):
        self.config = config
        self.fidelity_cost_list = config['fidelity_cost_list']

    def obtain_fidelity(self):
        return len(self.fidelity_cost_list) - 1


class Fidelity_Random():
    def __init__(self, config):
        self.config = config
        self.fidelity_cost_list = config['fidelity_cost_list']

    def obtain_fidelity(self):
        return np.random.choice(len(self.fidelity_cost_list), size=1)[0]


def find_best_fidelity(capacity, fidelity_cost_list):
    for i in range(len(fidelity_cost_list)):
        if capacity < fidelity_cost_list[i + 1]:
            return i
