import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DiBSCausalGraph():
    def __init__(self, config, graph, adj_weights_matrix):
        self.config = config

        self.graph = graph
        self.adj_weights_matrix = adj_weights_matrix

    def get_topological_graph(self):
        return self.graph

    def pass_forward(self, base_samples):
        masked_param = np.multiply(self.graph, self.adj_weights_matrix)
        y = np.matmul(base_samples, masked_param)

        return y


class DiBS(nn.Module):
    def __init__(self, config):
        super(DiBS, self).__init__()

        self.node_num = config['node_num']
        self.k = config['model_param']['k']
        self.lam = config['model_param']['lambda']
        self.lr = config['model_param']['lr']
        self.device = config['device']

        self.theta_mu = nn.Parameter(torch.randn(size=(self.node_num, self.node_num)))
        self.theta_sigma = nn.Parameter(torch.randn(size=(self.node_num, self.node_num)))
        self.S_mu = nn.Parameter(torch.randn(size=(self.k, self.node_num)))
        self.S_sigma = nn.Parameter(torch.randn(size=(self.k, self.node_num)))
        self.T_mu = nn.Parameter(torch.randn(size=(self.k, self.node_num)))
        self.T_sigma = nn.Parameter(torch.randn(size=(self.k, self.node_num)))

        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

    def get_mu_pack(self):
        return [self.theta_mu, self.S_mu, self.T_mu]

    def get_sigma_pack(self):
        return [self.theta_sigma, self.S_sigma, self.T_sigma]

    def sample_params(self):
        epsilon_theta = torch.randn(size=(self.node_num, self.node_num), requires_grad=False, device=self.device)
        epsilon_S = torch.randn(size=(self.k, self.node_num), requires_grad=False, device=self.device)
        epsilon_T = torch.randn(size=(self.k, self.node_num), requires_grad=False, device=self.device)

        function_param = self.theta_mu + epsilon_theta * self.theta_sigma
        S = self.S_mu + epsilon_S * self.S_sigma
        T = self.T_mu + epsilon_T * self.T_sigma

        return [function_param, S, T]

    def sample_graph(self):
        epsilon_theta = torch.randn(size=(self.node_num, self.node_num), requires_grad=False, device=self.device)
        epsilon_S = torch.randn(size=(self.k, self.node_num), requires_grad=False, device=self.device)
        epsilon_T = torch.randn(size=(self.k, self.node_num), requires_grad=False, device=self.device)

        function_param = self.theta_mu + epsilon_theta * self.theta_sigma
        S = self.S_mu + epsilon_S * self.S_sigma
        T = self.T_mu + epsilon_T * self.T_sigma

        tmp = torch.sigmoid(torch.matmul(S.T, T))
        nan_mask = tmp != tmp
        tmp[nan_mask] = 1.0

        adj_matrix = torch.bernoulli(tmp)

        return adj_matrix.detach(), function_param.detach()

    def sample_graph_numpy(self):
        adj_matrix, function_param = self.sample_graph()
        return adj_matrix.cpu().numpy(), function_param.cpu().numpy()

    def obtain_graph(self):
        epsilon_theta = torch.randn(size=(self.node_num, self.node_num), requires_grad=False, device=self.device)
        epsilon_S = torch.randn(size=(self.k, self.node_num), requires_grad=False, device=self.device)
        epsilon_T = torch.randn(size=(self.k, self.node_num), requires_grad=False, device=self.device)

        function_param = self.theta_mu + epsilon_theta * self.theta_sigma
        S = self.S_mu + epsilon_S * self.S_sigma
        T = self.T_mu + epsilon_T * self.T_sigma

        adj_matrix = torch.sigmoid(
            torch.matmul(S.T, T) + torch.tensor(np.random.logistic(size=(self.node_num, self.node_num)),
                                                requires_grad=False, device=self.device))

        return adj_matrix, function_param

    def update(self, samples_batch, j_batch, v_batch, m_batch):
        x = torch.tensor(samples_batch, requires_grad=False).to(self.device)

        adj_matrix, function_param = self.obtain_graph()
        masked_param = adj_matrix * function_param
        y = torch.matmul(x, masked_param)

        loss = F.mse_loss(x, y) + self.lam * (torch.trace(torch.matrix_exp(adj_matrix * adj_matrix)) - self.node_num)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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

    def calculate_p(self, y, graph, j, v):
        y_base = 1.0 * y
        adj_matrix, function_param = graph

        mask = torch.ones(size=(self.node_num, self.node_num), device=self.device) * 1.0 - \
               torch.eye(self.node_num, device=self.device)[j]
        adj_matrix = mask * adj_matrix
        masked_param = adj_matrix * function_param
        masked_param[j][j] = 1.0
        y[:, j] = v

        y_ = torch.matmul(y.to(self.device), masked_param.to(self.device))

        score = F.mse_loss(y_base.to(self.device), y_)

        return 1 / (1 + score.cpu())  # Approximate for p with 1 / (1+mse).
