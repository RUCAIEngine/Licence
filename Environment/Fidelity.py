import numpy as np


class Fidelity():
    def __init__(self, config):
        self.fidelity_noise_list = config['fidelity_noise_list']
        self.fidelity_cost_list = config['fidelity_cost_list']

    def pass_fidelity(self, m, sample):
        return sample + np.random.normal(0.0, self.fidelity_noise_list[m], size=sample.shape)
