import networkx as nx
import numpy as np
import graphical_models
import igraph as ig


def matrix_poly(matrix, d):
    x = np.eye(d) + matrix / d
    return np.linalg.matrix_power(x, d)


def exp_acyclic_function(E, n):
    exp_E = matrix_poly(E, n)
    h_A = np.trace(exp_E) - n
    return h_A


def num_mec(graph):
    dag = graphical_models.DAG.from_nx(graph)
    skeleton = dag.cpdag()
    all_dags = skeleton.all_dags()
    return len(all_dags)


def check_loop(graph):
    if graph is None:
        return True
    if exp_acyclic_function(nx.to_numpy_array(graph), len(graph.nodes)) == 0:  # No loop
        return False
    else:
        return True


def check_single_mec(graph):
    if graph is None:
        return True

    if num_mec(graph) >= 2:
        return False
    else:
        return True


class NoiseSampler():
    def __init__(self, config):
        self.mu = config['dataset_param']['noise_mu']
        self.sigma = config['dataset_param']['noise_sigma']

    def sample(self, size):
        return np.random.normal(loc=self.mu, scale=self.sigma, size=size)


class SFGraph():
    def __init__(self, config):
        self.config = config
        self.node_num = config['node_num']
        self.exp_edges = config['dataset_param']['exp_edges']

        self.graph = self.build_graph()
        self.function_params = self.build_function()
        self.noise_sampler = NoiseSampler(config)

        self.topological_order = [node for node in nx.topological_sort(self.graph)]

    def get_topological_graph(self):
        return self.graph

    def build_graph(self):

        graph = None
        while check_loop(graph) or check_single_mec(graph):
            perm = np.random.permutation(self.node_num).tolist()
            adj_m = ig.Graph.Barabasi(n=self.node_num, m=self.exp_edges, directed=True).permute_vertices(
                perm).get_adjacency()
            graph = nx.DiGraph(np.array(adj_m.data))

        return graph

    def build_function(self):
        # We implement the true causal graph with linear-SEM with ANM model.

        weights = np.random.uniform(-2, 2, size=len(self.graph.edges))
        adj_weights_matrix = np.zeros(shape=(self.node_num, self.node_num))
        for index, (s, t) in enumerate(self.graph.edges):
            adj_weights_matrix[s, t] = weights[index]

        return adj_weights_matrix

    def observation_sample(self, num):
        samples = np.zeros(shape=(self.node_num, num))
        for node in self.topological_order:
            values = self.noise_sampler.sample(size=num)
            for par in list(self.graph.predecessors(node)):
                values += samples[par] * self.function_params[par][node]
            samples[node] = values
        return samples.T

    def single_sample(self, j, v):
        sample = np.zeros(shape=(self.node_num,))
        for node in self.topological_order:
            values = self.noise_sampler.sample(size=(1,))
            if node != j:
                for par in list(self.graph.predecessors(node)):
                    values += sample[par] * self.function_params[par][node]
            else:
                values += v
            sample[node] = values
        return sample

    def batch_sample(self, intervention_list):
        samples = []
        for (j, v) in intervention_list:
            samples.append(self.single_sample(j, v))
        return np.array(samples)

    def pass_forward(self, base_samples):
        base_samples = base_samples.T
        output = np.zeros_like(base_samples)
        for node in self.topological_order:
            values = self.noise_sampler.sample(size=base_samples.shape[-1])
            for par in list(self.graph.predecessors(node)):
                values += base_samples[par] * self.function_params[par][node]
            output[node] = values
        return output.T
