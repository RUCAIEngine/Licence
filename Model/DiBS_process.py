import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class Toolkits():
    def __init__(self, config):
        self.config = config
        self.node_num = config['node_num']
        self.k = config['model_param']['k']
        self.device = config['device']

    def tensor_cal_single(self, list, func):
        output_list = []
        for index in range(len(list)):
            output_list.append(func(list[index]))
        return output_list

    def tensor_cal_double(self, list_1, list_2, func):
        output_list = []
        for index in range(len(list_1)):
            output_list.append(func(list_1[index], list_2[index]))
        return output_list

    def tensor_cal_triple(self, list_1, list_2, list_3, func):
        output_list = []
        for index in range(len(list_1)):
            output_list.append(func(list_1[index], list_2[index], list_3[index]))
        return output_list

    def sample_epsilon_tensor(self):
        epsilon_theta = torch.randn(size=(self.node_num, self.node_num), requires_grad=False, device=self.device)
        epsilon_S = torch.randn(size=(self.k, self.node_num), requires_grad=False, device=self.device)
        epsilon_T = torch.randn(size=(self.k, self.node_num), requires_grad=False, device=self.device)

        return [epsilon_theta, epsilon_S, epsilon_T]

    def calculate_log_p(self, y, j, v, graph):
        y_base = 1.0 * y
        adj_matrix, function_param = graph

        mask = torch.ones(size=(self.node_num, self.node_num), device=self.device) * 1.0 - \
               torch.eye(self.node_num, device=self.device)[j]
        adj_matrix = mask * adj_matrix
        masked_param = adj_matrix * function_param
        masked_param[j][j] = masked_param[j][j] - masked_param[j][j] + 1.0
        y = y.clone()
        y[:, j] = v

        y_ = torch.matmul(y.to(self.device), masked_param.to(self.device))

        gauss_epsilon = self.config['model_param']['gauss_epsilon']
        mse = F.mse_loss(y.to(self.device), y_)
        score = - self.node_num * np.log(gauss_epsilon * np.sqrt(2 * np.pi)) - 0.5 / gauss_epsilon / gauss_epsilon * mse

        return score

    def calculate_h(self, adj_matrix):
        h = torch.trace(torch.matrix_exp(adj_matrix * adj_matrix)) - self.node_num + torch.norm(adj_matrix) * \
            self.config['model_param']['lambda_h']
        return h

    def sample_interventional_samples(self, adj_matrix, function_param, j, v, num):
        mask = torch.ones(size=(self.node_num, self.node_num), device=self.device) * 1.0 - \
               torch.eye(self.node_num, device=self.device)[j]
        adj_matrix = mask * adj_matrix
        masked_param = adj_matrix * function_param
        masked_param[j][j] = 1.0
        base_sample = torch.randn(size=(num, self.node_num), device=self.device)
        base_sample[:, j] = v
        y = torch.matmul(base_sample, masked_param)
        return y.cpu()

    def sample_interventional_samples_by_grad(self, adj_matrix, function_param, j, v, num):
        mask = torch.ones(size=(self.node_num, self.node_num), device=self.device) * 1.0 - \
               torch.eye(self.node_num, device=self.device)[j]
        adj_matrix = mask * adj_matrix
        masked_param = adj_matrix * function_param
        masked_param[j][j] = masked_param[j][j] - masked_param[j][j] + 1.0
        base_sample = torch.randn(size=(num, self.node_num), device=self.device)
        base_sample[:, j] = v
        y = torch.matmul(base_sample, masked_param)
        return y


class Q_process(nn.Module):
    def __init__(self, config, mid):
        super(Q_process, self).__init__()

        self.config = config
        self.mid = mid  # [0, 1, ..., M-1]: q ( phi_(m+1) | phi_m )

        self.node_num = config['node_num']
        self.k = config['model_param']['k']

        self.w_theta = nn.Parameter(torch.randn(size=(self.node_num, self.node_num)))
        self.b_theta = nn.Parameter(torch.randn(size=(self.node_num, self.node_num)))
        self.w_S = nn.Parameter(torch.randn(size=(self.k, self.node_num)))
        self.b_S = nn.Parameter(torch.randn(size=(self.k, self.node_num)))
        self.w_T = nn.Parameter(torch.randn(size=(self.k, self.node_num)))
        self.b_T = nn.Parameter(torch.randn(size=(self.k, self.node_num)))

    def get_w_pack(self):
        return [self.w_theta, self.w_S, self.w_T]

    def get_b_pack(self):
        return [self.b_theta, self.b_S, self.b_T]


class P_process(nn.Module):
    def __init__(self, config, mid):
        super(P_process, self).__init__()

        self.config = config
        self.device = config['device']

        self.node_num = config['node_num']
        self.k = config['model_param']['k']
        self.mid = mid

        self.w_theta = torch.ones(size=(self.node_num, self.node_num), device=self.device)
        self.b_theta = torch.zeros(size=(self.node_num, self.node_num), device=self.device)
        self.w_S = torch.ones(size=(self.k, self.node_num), device=self.device)
        self.b_S = torch.zeros(size=(self.k, self.node_num), device=self.device)
        self.w_T = torch.ones(size=(self.k, self.node_num), device=self.device)
        self.b_T = torch.zeros(size=(self.k, self.node_num), device=self.device)

    def get_w_pack(self):
        return [self.w_theta, self.w_S, self.w_T]

    def get_b_pack(self):
        return [self.b_theta, self.b_S, self.b_T]
