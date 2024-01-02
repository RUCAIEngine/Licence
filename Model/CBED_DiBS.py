import networkx as nx

from Model.utils import ReplayBuffer
from Model.utils import Fidelity_Min, Fidelity_Max, Fidelity_Random
from Model.utils import find_best_fidelity
from bayes_opt import BayesianOptimization

from Model.DiBS import DiBS, DiBSCausalGraph
import numpy as np


class CBED():
    def __init__(self, config, model):
        self.config = config
        self.model = model

        self.node_num = config['node_num']
        self.current_j = 1

        self.sample_co_num = 2
        self.sample_m_num = 5
        self.sample_cin_num = 2

    def MI_score(self, v):
        pool = []
        graph_list = []
        for i in range(self.sample_co_num):
            adj_matrix, function_param = self.model.sample_graph()
            graph_list.append((adj_matrix, function_param))
            samples = self.model.sample_interventional_samples(adj_matrix, function_param, self.current_j, v,
                                                               self.sample_m_num)
            pool.append(samples)
        sec_graph_list = []
        for i in range(self.sample_cin_num):
            adj_matrix, function_param = self.model.sample_graph()
            sec_graph_list.append((adj_matrix, function_param))

        avg_co = 0
        for i in range(self.sample_co_num):
            h_y_theta = np.log(self.model.calculate_p(pool[i], graph_list[i], self.current_j, v))
            avg_cin = 0
            for j in range(self.sample_cin_num):
                h_y = np.log(self.model.calculate_p(pool[i], sec_graph_list[j], self.current_j, v))
                avg_cin += h_y
            h_y = avg_cin / self.sample_cin_num
            avg_co += h_y - h_y_theta
        avg_co /= self.sample_co_num
        return avg_co

    def calculate_best_value(self):
        bayesian_opt = BayesianOptimization(
            f=self.MI_score,
            pbounds={'v': (-3, 3)},
            verbose=0,
            allow_duplicate_points=True
        )
        bayesian_opt.maximize(n_iter=10, init_points=5)
        return bayesian_opt.max['target'], bayesian_opt.max['params']['v']

    def acquire(self):
        best_j, best_v, best_target = None, -10000, -10000
        for j in range(self.node_num):
            self.current_j = j
            target, v = self.calculate_best_value()
            if target > best_target:
                best_v = v
                best_j = j
                best_target = target
        return best_j, best_v


class CBED_DiBS():
    def __init__(self, config):
        self.config = config
        self.node_num = config['node_num']

        self.training_batch_size = config['model_param']['training_batch_size']
        self.training_epoch = config['model_param']['training_epoch']

        self.replay_buffer = ReplayBuffer(config)
        self.dibs = DiBS(config)
        self.dibs.to(config['device'])
        self.cbed = CBED(config, self.dibs)
        self.fidelity_selector = eval('Fidelity_%s' % config['fidelity_strategy'])(config)

    def observational_train(self, observational_data):
        for x in observational_data:
            self.replay_buffer.add_obs(x)

        for i in range(self.training_epoch):
            for j in range(int(self.replay_buffer.total() / self.training_batch_size) + 1):
                samples_batch, j_batch, v_batch, m_batch = self.replay_buffer.sample_batch(self.training_batch_size)
                self.dibs.update(samples_batch, j_batch, v_batch, m_batch)

    def generate_single_query(self):
        j, v = self.cbed.acquire()
        m = self.fidelity_selector.obtain_fidelity()
        return (j, v, m)

    def generate_batch_query(self):
        batch_query = []
        consumption = 0
        while consumption + self.config['fidelity_cost_list'][0] <= self.config['acq_param']['acq_batch_budge']:
            j, v = self.cbed.acquire()
            m = self.fidelity_selector.obtain_fidelity()
            if consumption + self.config['fidelity_cost_list'][m] > self.config['acq_param']['acq_batch_budge']:
                m = find_best_fidelity(self.config['acq_param']['acq_batch_budge'] - consumption,
                                       self.config['fidelity_cost_list'])
            consumption += self.config['fidelity_cost_list'][m]
            batch_query.append((j, v, m))
        return batch_query

    def single_update(self, single_query, single_interventional_data):
        j, v, m = single_query
        self.replay_buffer.add_int(single_interventional_data, j, v, m)

        for i in range(self.training_epoch):
            for j in range(int(self.replay_buffer.total() / self.training_batch_size) + 1):
                samples_batch, j_batch, v_batch, m_batch = self.replay_buffer.sample_batch(self.training_batch_size)
                self.dibs.update(samples_batch, j_batch, v_batch, m_batch)

    def batch_update(self, batch_query, batch_interventional_data):
        for (j, v, m), x in zip(batch_query, batch_interventional_data):
            self.replay_buffer.add_int(x, j, v, m)
        for i in range(self.training_epoch):
            for j in range(int(self.replay_buffer.total() / self.training_batch_size) + 1):
                samples_batch, j_batch, v_batch, m_batch = self.replay_buffer.sample_batch(self.training_batch_size)
                self.dibs.update(samples_batch, j_batch, v_batch, m_batch)

    def output_graph(self):
        adj_matrix, function_param = self.dibs.sample_graph_numpy()
        return DiBSCausalGraph(self.config, adj_matrix, function_param)
