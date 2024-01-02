import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from Model.utils import ReplayBuffer, find_best_fidelity
from Model.utils import Fidelity_Min, Fidelity_Max, Fidelity_Random
from Model.DiBS_process import Q_process, P_process, Toolkits
from Model.DiBS import DiBS, DiBSCausalGraph

from bayes_opt import BayesianOptimization

from copy import deepcopy


class Licence(nn.Module):
    def __init__(self, config):
        super(Licence, self).__init__()
        self.config = config
        self.toolkit = Toolkits(config)
        self.device = config['device']

        self.node_num = config['node_num']
        self.fidelity_num = len(config['fidelity_cost_list'])
        self.fidelity_cost_list = config['fidelity_cost_list']
        self.training_batch_size = config['model_param']['training_batch_size']
        self.training_epoch = config['model_param']['training_epoch']
        self.lr = config['model_param']['lr']
        self.lambda1 = config['model_param']['lambda1']
        self.lambda2 = config['model_param']['lambda2']
        self.lambda3 = config['model_param']['lambda3']
        self.lambda4 = config['model_param']['lambda4']

        self.n_s = config['model_param']['n_s']
        self.n_d = config['model_param']['n_d']
        self.n_e = config['model_param']['n_e']
        self.beta = config['model_param']['beta']
        self.k_1 = config['model_param']['k_1']
        self.l_1 = config['model_param']['l_1']
        self.c_1 = config['model_param']['c_1']
        self.k_2 = config['model_param']['k_2']
        self.l_2 = config['model_param']['l_2']
        self.c_2 = config['model_param']['c_2']
        self.k_3 = config['model_param']['k_3']
        self.l_3 = config['model_param']['l_3']
        self.c_3 = config['model_param']['c_3']

        self.replay_buffer = ReplayBuffer(config)
        self.base_dibs = DiBS(config)
        self.q_process_list = [Q_process(config, i) for i in range(self.fidelity_num)]
        self.p_process_list = [P_process(config, i) for i in range(self.fidelity_num)]

        for i in range(self.fidelity_num):
            self.register_module('q_%d' % i, self.q_process_list[i])

        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

        self.current_m = 0
        self.current_j = 0

    def sample_base_params(self):
        return self.base_dibs.sample_params()

    def obtain_params(self, m, sample_epsilon=None):
        bar_w_pack_list = [[self.q_process_list[0].w_theta, self.q_process_list[0].w_S, self.q_process_list[0].w_T]]
        bar_b_pack_list = [[self.q_process_list[0].b_theta, self.q_process_list[0].b_S, self.q_process_list[0].b_T]]
        sigma_pack_list = [self.base_dibs.get_sigma_pack()]
        for i in range(1, m + 1):
            w_i = self.q_process_list[i].get_w_pack()
            b_i = self.q_process_list[i].get_b_pack()
            bar_w_pack = self.toolkit.tensor_cal_double(bar_w_pack_list[i - 1], w_i, func=lambda x, y: torch.mul(x, y))
            bar_b_pack = self.toolkit.tensor_cal_triple(bar_b_pack_list[i - 1], w_i, b_i,
                                                        func=lambda x, y, z: torch.mul(x, y) + z)
            bar_w_pack_list.append(bar_w_pack)
            bar_b_pack_list.append(bar_b_pack)

            if i == 1 or i == 2:
                sigma_pack = sigma_pack_list[i - 1]
            else:
                coef = self.toolkit.tensor_cal_single(self.q_process_list[i - 2].get_w_pack(),
                                                      lambda x: torch.sqrt(torch.pow(x, 2) + 1))
                sigma_pack = self.toolkit.tensor_cal_double(coef, sigma_pack_list[i - 1], lambda x, y: torch.mul(x, y))
            sigma_pack_list.append(sigma_pack)

        if m + 1 == 1 or m + 1 == 2:
            sigma_pack = sigma_pack_list[-1]
        else:
            coef = self.toolkit.tensor_cal_single(self.q_process_list[m + 1 - 2].get_w_pack(),
                                                  lambda x: torch.sqrt(torch.pow(x, 2) + 1))
            sigma_pack = self.toolkit.tensor_cal_double(coef, sigma_pack_list[m + 1 - 1], lambda x, y: torch.mul(x, y))
        sigma_pack_list.append(sigma_pack)

        base_params = self.sample_base_params()

        sample_mu = self.toolkit.tensor_cal_triple(bar_w_pack_list[-1], base_params, bar_b_pack_list[-1],
                                                   func=lambda x, y, z: torch.mul(x, y) + z)
        coef = self.toolkit.tensor_cal_single(self.q_process_list[m + 1 - 1].get_w_pack(),
                                              lambda x: torch.sqrt(torch.pow(x, 2) + 1))
        sample_sigma = self.toolkit.tensor_cal_double(coef, sigma_pack_list[-1], lambda x, y: torch.mul(x, y))

        if not sample_epsilon:
            sample_epsilon = self.toolkit.sample_epsilon_tensor()

        graph_params = self.toolkit.tensor_cal_triple(sample_mu, sample_sigma, sample_epsilon,
                                                      func=lambda x, y, z: torch.mul(y, z) + x)
        return graph_params

    def obtain_graph(self, m, sample_epsilon=None):
        theta_param, S_param, T_param = self.obtain_params(m, sample_epsilon)

        adj_matrix = torch.sigmoid(
            torch.matmul(S_param.T, T_param) + torch.tensor(np.random.logistic(size=(self.node_num, self.node_num)),
                                                            requires_grad=False, device=self.device,
                                                            dtype=torch.float32))
        return adj_matrix, theta_param

    def obtain_graph_from_params(self, graph_params):
        theta_param, S_param, T_param = graph_params
        adj_matrix = torch.sigmoid(
            torch.matmul(S_param.T, T_param) + torch.tensor(np.random.logistic(size=(self.node_num, self.node_num)),
                                                            requires_grad=False, device=self.device,
                                                            dtype=torch.float32))
        return adj_matrix, theta_param

    def optimize(self, samples_batch, j_batch, v_batch, m_batch):
        loss = torch.tensor(0.0, requires_grad=True, device=self.device)

        # Part 1 loss
        for index in range(samples_batch.shape[0]):
            x, j, v, m = torch.tensor(samples_batch[index], device=self.device, dtype=torch.float32).view(1, -1), \
            j_batch[index], v_batch[
                index], m_batch[index]
            for i in range(self.n_s):
                graph = self.obtain_graph(m)
                log_p = self.toolkit.calculate_log_p(x, j, v, graph)
                loss = loss + log_p
        loss = -1.0 * loss / self.n_s / samples_batch.shape[0] * self.lambda1
        # print('loss1:',loss)

        # Part 2 loss
        loss_part2 = torch.tensor(0.0, requires_grad=True, device=self.device)
        base_params = self.sample_base_params()
        q_params_sigma = base_params
        p_params_sigma = base_params
        for index in range(self.fidelity_num):
            q_params_mu = self.toolkit.tensor_cal_triple(self.q_process_list[index].get_w_pack(), base_params,
                                                         self.q_process_list[index].get_b_pack(),
                                                         func=lambda x, y, z: torch.mul(x, y) + z)
            p_params_mu = self.toolkit.tensor_cal_triple(self.p_process_list[index].get_w_pack(), base_params,
                                                         self.p_process_list[index].get_b_pack(),
                                                         func=lambda x, y, z: torch.mul(x, y) + z)
            if index + 1 <= 2:
                pass
            else:
                q_coef = self.toolkit.tensor_cal_single(self.q_process_list[index - 2].get_w_pack(),
                                                        lambda x: torch.sqrt(torch.pow(x, 2) + 1))
                q_params_sigma = self.toolkit.tensor_cal_double(q_coef, q_params_sigma, lambda x, y: torch.mul(x, y))

                p_coef = self.toolkit.tensor_cal_single(self.p_process_list[index - 2].get_w_pack(),
                                                        lambda x: torch.sqrt(torch.pow(x, 2) + 1))
                p_params_sigma = self.toolkit.tensor_cal_double(p_coef, p_params_sigma, lambda x, y: torch.mul(x, y))

            term1 = self.toolkit.tensor_cal_double(p_params_sigma, q_params_sigma,
                                                   lambda x, y: torch.log(torch.norm(x) / torch.norm(y)))
            term1 = term1[0] + term1[1] + term1[2]
            term2 = - self.node_num * self.node_num - self.base_dibs.k * self.node_num - self.base_dibs.k * self.node_num
            term3 = self.toolkit.tensor_cal_double(p_params_sigma, q_params_sigma,
                                                   lambda x, y: torch.sum(torch.mul(1 / x, y)))
            term3 = term3[0] + term3[1] + term3[2]
            term4_dif = self.toolkit.tensor_cal_double(p_params_mu, q_params_mu, lambda x, y: x - y)
            term4 = self.toolkit.tensor_cal_double(term4_dif, p_params_sigma,
                                                   lambda x, y: torch.sum(torch.mul(torch.mul(x, x), 1 / y)))
            term4 = term4[0] + term4[1] + term4[2]
            loss_part2 = loss_part2 - 0.5 * (term1 + term2 + term3 + term4)
        loss_part2 = loss_part2 / self.fidelity_num
        # print('loss2:',-loss_part2)
        loss = loss - loss_part2 * self.lambda2

        # Part3 loss
        loss_part3 = torch.tensor(0.0, requires_grad=True, device=self.device)
        for i in range(self.n_d):
            sample_epsilon = self.toolkit.sample_epsilon_tensor()
            for m in range(self.fidelity_num):
                graph_params = self.obtain_params(m, sample_epsilon)
                for l in range(self.n_e):
                    adj_matrix, theta_param = self.obtain_graph_from_params(graph_params)
                    h = self.toolkit.calculate_h(adj_matrix)
                    loss_part3 = loss_part3 - h
        loss_part3 = loss_part3 / self.n_d / self.fidelity_num / self.n_e * self.beta
        # print('loss3:', -loss_part3)
        loss = loss - loss_part3 * self.lambda3
        # print('total_loss:',loss)
        # raise

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.p_process_list = deepcopy(self.q_process_list)
        return loss.detach()

    def optimize_batch(self, samples_batch, j_batch, v_batch, m_batch):
        loss = torch.tensor(0.0, requires_grad=True, device=self.device)

        # Part 1 loss
        for index in range(samples_batch.shape[0]):
            x, j, v, m = torch.tensor(samples_batch[index], device=self.device, dtype=torch.float32).view(1, -1), \
            j_batch[index], v_batch[
                index], m_batch[index]
            for i in range(self.n_s):
                graph = self.obtain_graph(m)
                log_p = self.toolkit.calculate_log_p(x, j, v, graph)
                loss = loss + log_p
        loss = -1.0 * loss / self.n_s / samples_batch.shape[0] * self.lambda1
        # print('loss1:',loss)

        # Part 2 loss
        loss_part2 = torch.tensor(0.0, requires_grad=True, device=self.device)
        base_params = self.sample_base_params()
        q_params_sigma = base_params
        p_params_sigma = base_params
        for index in range(self.fidelity_num):
            q_params_mu = self.toolkit.tensor_cal_triple(self.q_process_list[index].get_w_pack(), base_params,
                                                         self.q_process_list[index].get_b_pack(),
                                                         func=lambda x, y, z: torch.mul(x, y) + z)
            p_params_mu = self.toolkit.tensor_cal_triple(self.p_process_list[index].get_w_pack(), base_params,
                                                         self.p_process_list[index].get_b_pack(),
                                                         func=lambda x, y, z: torch.mul(x, y) + z)
            if index + 1 <= 2:
                pass
            else:
                q_coef = self.toolkit.tensor_cal_single(self.q_process_list[index - 2].get_w_pack(),
                                                        lambda x: torch.sqrt(torch.pow(x, 2) + 1))
                q_params_sigma = self.toolkit.tensor_cal_double(q_coef, q_params_sigma, lambda x, y: torch.mul(x, y))

                p_coef = self.toolkit.tensor_cal_single(self.p_process_list[index - 2].get_w_pack(),
                                                        lambda x: torch.sqrt(torch.pow(x, 2) + 1))
                p_params_sigma = self.toolkit.tensor_cal_double(p_coef, p_params_sigma, lambda x, y: torch.mul(x, y))

            term1 = self.toolkit.tensor_cal_double(p_params_sigma, q_params_sigma,
                                                   lambda x, y: torch.log(torch.norm(x) / torch.norm(y)))
            term1 = term1[0] + term1[1] + term1[2]
            term2 = - self.node_num * self.node_num - self.base_dibs.k * self.node_num - self.base_dibs.k * self.node_num
            term3 = self.toolkit.tensor_cal_double(p_params_sigma, q_params_sigma,
                                                   lambda x, y: torch.sum(torch.mul(1 / x, y)))
            term3 = term3[0] + term3[1] + term3[2]
            term4_dif = self.toolkit.tensor_cal_double(p_params_mu, q_params_mu, lambda x, y: x - y)
            term4 = self.toolkit.tensor_cal_double(term4_dif, p_params_sigma,
                                                   lambda x, y: torch.sum(torch.mul(torch.mul(x, x), 1 / y)))
            term4 = term4[0] + term4[1] + term4[2]
            loss_part2 = loss_part2 - 0.5 * (term1 + term2 + term3 + term4)
        loss_part2 = loss_part2 / self.fidelity_num
        # print('loss2:',-loss_part2)
        loss = loss - loss_part2 * self.lambda2

        # Part3 loss
        loss_part3 = torch.tensor(0.0, requires_grad=True, device=self.device)
        for i in range(self.n_d):
            sample_epsilon = self.toolkit.sample_epsilon_tensor()
            for m in range(self.fidelity_num):
                graph_params = self.obtain_params(m, sample_epsilon)
                for l in range(self.n_e):
                    adj_matrix, theta_param = self.obtain_graph_from_params(graph_params)
                    h = self.toolkit.calculate_h(adj_matrix)
                    loss_part3 = loss_part3 - h
        loss_part3 = loss_part3 / self.n_d / self.fidelity_num / self.n_e * self.beta
        # print('loss3:', -loss_part3)
        loss = loss - loss_part3 * self.lambda3

        # Part4 loss

        sample_ind = np.random.choice(samples_batch.shape[0], size=1)[0]
        x_tmp, j_tmp, v_tmp, m_tmp = samples_batch[sample_ind].reshape(1, -1), j_batch[sample_ind], v_batch[sample_ind], \
        m_batch[sample_ind]

        term1_all = torch.tensor(0.0, requires_grad=True, device=self.device)
        term2_all = torch.tensor(0.0, requires_grad=True, device=self.device)
        for i in range(self.k_3):
            graph1 = self.obtain_graph(m_tmp)
            adj_matrix1, function_param1 = graph1[0], graph1[1]
            for j in range(self.l_3):
                sample_2 = self.toolkit.sample_interventional_samples_by_grad(adj_matrix1, function_param1, j_tmp,
                                                                              v_tmp, 1)
                sample_3 = self.toolkit.sample_interventional_samples_by_grad(adj_matrix1, function_param1, j_tmp,
                                                                              v_tmp, 1)
                term1_log = torch.tensor(0.0, requires_grad=True, device=self.device)
                term2_log = torch.tensor(0.0, requires_grad=True, device=self.device)
                for l in range(self.c_3):
                    graph3 = self.obtain_graph(m_tmp)
                    adj_matrix3, function_param3 = graph3[0], graph3[1]
                    term1_log = term1_log + self.toolkit.calculate_log_p(sample_2, j_tmp, v_tmp, graph3)
                    tt = torch.concat([sample_2, sample_3], dim=-2)
                    term2_log = term2_log + self.toolkit.calculate_log_p(torch.concat([sample_2, sample_3], dim=-2),
                                                                         j_tmp, v_tmp, graph3)
                term1_all = term1_all + term1_log
                term2_all = term2_all + term2_log
        term1_all = term1_all / self.k_3 / self.l_3
        term2_all = term2_all / self.k_3 / self.l_3
        loss_part4 = (2 * term1_all - term2_all) * self.lambda4

        loss = loss - loss_part4
        # print('total_loss:',loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.p_process_list = deepcopy(self.q_process_list)
        return loss.detach()

    def observational_train(self, observational_data):
        for x in observational_data:
            self.replay_buffer.add_obs(x)
        for i in range(self.training_epoch):
            avg_ls = 0.0
            for j in range(int(self.replay_buffer.total() / self.training_batch_size) + 1):
                samples_batch, j_batch, v_batch, m_batch = self.replay_buffer.sample_batch(self.training_batch_size)
                if self.config['acq_param']['acq_type'] == 'single':
                    ls = self.optimize(samples_batch, j_batch, v_batch, m_batch)
                else:
                    ls = self.optimize_batch(samples_batch, j_batch, v_batch, m_batch)
                avg_ls += ls
            avg_ls /= int(self.replay_buffer.total() / self.training_batch_size) + 1
            # print(i, avg_ls)

    def inverse_sample(self, graph_params_cur, m, M):
        graph_params_cur = self.toolkit.tensor_cal_single(graph_params_cur, lambda x: x.detach())

        bar_w_pack_list = [[self.q_process_list[0].w_theta, self.q_process_list[0].w_S, self.q_process_list[0].w_T]]
        bar_b_pack_list = [[self.q_process_list[0].b_theta, self.q_process_list[0].b_S, self.q_process_list[0].b_T]]
        sigma_pack_list = [self.base_dibs.get_sigma_pack()]
        for i in range(1, M):
            w_i = self.q_process_list[i].get_w_pack()
            b_i = self.q_process_list[i].get_b_pack()
            bar_w_pack = self.toolkit.tensor_cal_double(bar_w_pack_list[i - 1], w_i, func=lambda x, y: torch.mul(x, y))
            bar_b_pack = self.toolkit.tensor_cal_triple(bar_b_pack_list[i - 1], w_i, b_i,
                                                        func=lambda x, y, z: torch.mul(x, y) + z)
            bar_w_pack_list.append(bar_w_pack)
            bar_b_pack_list.append(bar_b_pack)

            if i == 1 or i == 2:
                sigma_pack = sigma_pack_list[i - 1]
            else:
                coef = self.toolkit.tensor_cal_single(self.q_process_list[i - 2].get_w_pack(),
                                                      lambda x: torch.sqrt(torch.pow(x, 2) + 1))
                sigma_pack = self.toolkit.tensor_cal_double(coef, sigma_pack_list[i - 1], lambda x, y: torch.mul(x, y))
            sigma_pack_list.append(sigma_pack)

        if M == 1 or M == 2:
            sigma_pack = sigma_pack_list[-1]
        else:
            coef = self.toolkit.tensor_cal_single(self.q_process_list[M - 2].get_w_pack(),
                                                  lambda x: torch.sqrt(torch.pow(x, 2) + 1))
            sigma_pack = self.toolkit.tensor_cal_double(coef, sigma_pack_list[M - 1], lambda x, y: torch.mul(x, y))
        sigma_pack_list.append(sigma_pack)

        def inverse_forward(graph_params, m):
            m += 1  # recovery for 1 to M
            # Input: phi_m
            # Output: phi_(m-1)
            sigma_tmp = self.toolkit.tensor_cal_double(sigma_pack_list[m], self.q_process_list[m - 1].get_w_pack(),
                                                       func=lambda x, y: torch.mul(x, x) / (torch.mul(y, y) + 1))
            mu_tmp_1 = self.toolkit.tensor_cal_double(graph_params, self.q_process_list[m - 1].get_w_pack(),
                                                      func=lambda x, y: torch.mul(x, y))
            mu_tmp_23 = self.toolkit.tensor_cal_triple(self.base_dibs.sample_params(), bar_w_pack_list[m - 1],
                                                       bar_b_pack_list[m - 2], lambda x, y, z: torch.mul(x, y) + z)

            mu_tmp_4 = self.toolkit.tensor_cal_double(self.q_process_list[m - 1].get_b_pack(),
                                                      self.q_process_list[m - 1].get_w_pack(),
                                                      lambda x, y: torch.mul(x, y))
            mu_fz = self.toolkit.tensor_cal_triple(mu_tmp_1, mu_tmp_23, mu_tmp_4, lambda x, y, z: x + y - z)
            mu_tmp = self.toolkit.tensor_cal_double(mu_fz, self.q_process_list[m - 1].get_w_pack(),
                                                    func=lambda x, y: x / (torch.mul(y, y) + 1))

            epsilon = self.toolkit.sample_epsilon_tensor()
            new_params = self.toolkit.tensor_cal_triple(mu_tmp, sigma_tmp, epsilon, lambda x, y, z: x + torch.mul(y, z))

            return new_params

        for i in range(M - m):
            input_m = M - i - 1
            graph_params_cur = inverse_forward(graph_params_cur, input_m)
        return graph_params_cur

    def MI_cost_score(self, v):
        term1 = torch.tensor(0.0, device=self.device)

        graph_c_list = []
        for c in range(self.c_1):
            graph = self.obtain_graph(self.current_m)
            graph_c_list.append((graph[0].detach(), graph[1].detach()))

        for i in range(self.k_1):
            graph = self.obtain_graph(self.current_m)
            adj_matrix, function_param = graph[0].detach(), graph[1].detach()
            x_batch = self.toolkit.sample_interventional_samples(adj_matrix, function_param, self.current_j, v,
                                                                 self.l_1)
            for l in range(self.l_1):
                for c in range(self.c_1):
                    term1 += self.toolkit.calculate_log_p(x_batch, self.current_j, v, graph_c_list[c])
        term1 /= - self.fidelity_cost_list[self.current_m] * self.k_1 * self.l_1

        term2 = torch.tensor(0.0, device=self.device)

        for i in range(self.k_2):
            graph_params_M = self.obtain_params(self.current_m)
            graph_M = self.obtain_graph_from_params(graph_params_M)
            for l in range(self.l_2):
                graph_params_m = self.inverse_sample(graph_params_M, self.current_m, self.fidelity_num)
                graph_m = self.obtain_graph_from_params(graph_params_m)
                x_batch = self.toolkit.sample_interventional_samples(graph_m[0], graph_m[1], self.current_j, v,
                                                                     self.c_2)
                term2 += self.toolkit.calculate_log_p(x_batch, self.current_j, v, graph_M).detach()
        term2 /= self.fidelity_cost_list[self.current_m] * self.k_2 * self.l_2

        score = term1 + term2
        return score

    def acquire(self):
        def calculate_best_value():
            bayesian_opt = BayesianOptimization(
                f=self.MI_cost_score,
                pbounds={'v': (-3, 3)},
                verbose=0,
                allow_duplicate_points=True
            )
            bayesian_opt.maximize(n_iter=10, init_points=5)
            return bayesian_opt.max['target'], bayesian_opt.max['params']['v']

        best_j, best_v, best_m, best_target = None, -10000, None, -10000
        for m in range(self.fidelity_num):
            for j in range(self.node_num):
                self.current_m = m
                self.current_j = j
                target, v = calculate_best_value()
                if target > best_target:
                    best_v = v
                    best_j = j
                    best_m = m
                    best_target = target
        return best_j, best_v, best_m

    def generate_single_query(self):
        j, v, m = self.acquire()
        return (j, v, m)

    def generate_batch_query(self):
        batch_query = []
        consumption = 0
        while consumption + self.config['fidelity_cost_list'][0] <= self.config['acq_param']['acq_batch_budge']:
            j, v, m = self.acquire()
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
            avg_loss = 0.0
            for j in range(int(self.replay_buffer.total() / self.training_batch_size) + 1):
                samples_batch, j_batch, v_batch, m_batch = self.replay_buffer.sample_batch(self.training_batch_size)
                avg_loss += self.optimize(samples_batch, j_batch, v_batch, m_batch)
            avg_loss /= int(self.replay_buffer.total() / self.training_batch_size) + 1
            # print(i, avg_loss)

    def batch_update(self, batch_query, batch_interventional_data):
        for (j, v, m), x in zip(batch_query, batch_interventional_data):
            self.replay_buffer.add_int(x, j, v, m)
        for i in range(self.training_epoch):
            avg_loss = 0.0
            for j in range(int(self.replay_buffer.total() / self.training_batch_size) + 1):
                samples_batch, j_batch, v_batch, m_batch = self.replay_buffer.sample_batch(self.training_batch_size)
                avg_loss += self.optimize_batch(samples_batch, j_batch, v_batch, m_batch)
            avg_loss /= int(self.replay_buffer.total() / self.training_batch_size) + 1
            print(i, avg_loss)

    def output_graph(self):
        function_param, S, T = self.obtain_params(self.fidelity_num - 1)
        adj_matrix = torch.bernoulli(torch.sigmoid(torch.matmul(S.T, T)))
        adj_matrix, function_param = adj_matrix.detach().numpy(), function_param.detach().numpy()
        return DiBSCausalGraph(self.config, adj_matrix, function_param)

    def output_graph_details(self, m):
        function_param, S, T = self.obtain_params(m)
        adj_matrix = torch.sigmoid(torch.matmul(S.T, T))
        return adj_matrix.detach().numpy()
