import numpy as np
import graphical_models
import networkx as nx


def matrix_poly(matrix, d):
    x = np.eye(d) + matrix / d
    return np.linalg.matrix_power(x, d)


def exp_acyclic_function(E, n):
    exp_E = matrix_poly(E, n)
    h_A = np.trace(exp_E) - n
    return h_A


def check_loop(graph):
    if graph is None:
        return True
    if exp_acyclic_function(nx.to_numpy_array(graph), len(graph.nodes)) == 0:  # No loop
        return False
    else:
        return True


class RandomCausalGraph():
    def __init__(self, graph, adj_weights_matrix):
        self.graph = graph
        self.adj_weights_matrix = adj_weights_matrix

        self.topological_order = [node for node in nx.topological_sort(self.graph)]

    def get_topological_graph(self):
        return nx.to_numpy_array(self.graph, dtype=int)

    def pass_forward(self, base_samples):
        base_samples = base_samples.T
        output = np.zeros_like(base_samples)
        for node in self.topological_order:
            values = np.random.normal(size=base_samples.shape[-1])
            for par in list(self.graph.predecessors(node)):
                values += base_samples[par] * self.adj_weights_matrix[par][node]
            output[node] = values
        return output.T


class Random():
    def __init__(self, config):
        self.config = config
        self.node_num = config['node_num']

    def observational_train(self, observational_data):
        pass

    def generate_single_query(self):
        j = np.random.randint(self.config['node_num'])
        v = np.random.normal()
        m = np.random.randint(len(self.config['fidelity_noise_list']))
        return (j, v, m)

    def generate_batch_query(self):
        batch_query = []
        consumption = 0
        while consumption + self.config['fidelity_cost_list'][0] <= self.config['acq_param']['acq_batch_budge']:
            j = np.random.randint(self.config['node_num'])
            v = np.random.normal()
            m = np.random.randint(len(self.config['fidelity_noise_list']))
            if consumption + self.config['fidelity_cost_list'][m] <= self.config['acq_param']['acq_batch_budge']:
                consumption += self.config['fidelity_cost_list'][m]
                batch_query.append((j, v, m))
        return batch_query

    def single_update(self, single_query, single_interventional_data):
        pass

    def batch_update(self, batch_query, batch_interventional_data):
        pass

    def random_generate_graph(self):
        # exp_edges = np.random.randint(self.node_num)
        exp_edges = 1
        p = float(exp_edges) / (self.node_num - 1)

        graph = None
        while check_loop(graph):
            if exp_edges <= 2:
                graph = nx.generators.fast_gnp_random_graph(self.node_num, p, directed=True)
            else:
                graph = nx.generators.gnp_random_graph(self.node_num, p, directed=True)

                # We implement the true causal graph with linear-SEM with ANM model.

        weights = np.random.uniform(-2, 2, size=len(graph.edges))
        adj_weights_matrix = np.zeros(shape=(self.node_num, self.node_num))
        for index, (s, t) in enumerate(graph.edges):
            adj_weights_matrix[s, t] = weights[index]

        return graph, adj_weights_matrix

    def output_graph(self):
        graph, adj_weights_matrix = self.random_generate_graph()
        return RandomCausalGraph(graph, adj_weights_matrix)
