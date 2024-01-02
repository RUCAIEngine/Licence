from Environment.Fidelity import Fidelity
from Environment.utils import create_causal_graph
import numpy as np


class Environment():
    def __init__(self, config):
        self.config = config
        self.true_causal_graph = create_causal_graph(config)
        self.fidelity = Fidelity(config)

    def initial_observational_samples(self):
        return self.true_causal_graph.observation_sample(self.config['obs_data_num'])

    def single_acquire(self, query):
        j, v, m = query
        return self.fidelity.pass_fidelity(m, self.true_causal_graph.single_sample(j, v))

    def batch_acquire(self, queries):
        batch_sample = []
        for (j, v, m) in queries:
            s = self.fidelity.pass_fidelity(m, self.true_causal_graph.single_sample(j, v))
            batch_sample.append(s)
        return np.array(batch_sample)
