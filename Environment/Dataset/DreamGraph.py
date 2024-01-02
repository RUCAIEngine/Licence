import os
from xml.dom import minidom
import numpy as np
import networkx as nx
import subprocess
import pandas as pd
import shutil


def check_dir_and_make(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_network(xml):
    xmldoc = minidom.parse(str(xml))

    nodes = []
    var2id = {}
    for i, node in enumerate(xmldoc.getElementsByTagName('species')):
        name = node.attributes.get('id').value
        if 'void' not in name:
            nodes.append(name)
            var2id[name] = i

    A = np.zeros((len(nodes), len(nodes)))

    for node in xmldoc.getElementsByTagName('reaction'):
        # child
        child = node.getElementsByTagName('listOfProducts')[0].getElementsByTagName('speciesReference')[
            0].attributes.get('species').value

        # parents
        for parent in node.getElementsByTagName('modifierSpeciesReference'):
            _from = var2id[parent.attributes.get('species').value]
            _to = var2id[child]
            A[_from, _to] = 1

    return nodes, var2id, A


class DreamGraph():
    def __init__(self, config):
        self.config = config

        self.node_num = config['node_num']

        self.dream_path = 'Environment/Dataset/DreamFiles'
        # self.software_path = '%s/gnw-3.1b.jar' % self.dream_path
        # self.setting_path = '%s/settings.txt' % self.dream_path
        self.name = 'InSilicoSize%d-%s' % (self.node_num, config['dataset_param']['name'])
        self.file_path = '%s/configurations/%s.xml' % (self.dream_path, self.name)

        self.nodes, self.var2id, adj_matrix = get_network(self.file_path)
        self.id2var = {v: k for k, v in self.var2id.items()}

        self.graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

        self.topological_order = [node for node in nx.topological_sort(self.graph)]

    def get_topological_graph(self):
        return self.graph

    def observation_sample(self, num):
        software_path = '../../gnw-3.1b.jar'
        setting_path = '../../settings.txt'
        network_path = '../../configurations/%s.xml' % self.name
        cmd = 'java -jar %s -c %s --input-net %s --output-net-format=1 --simulate' % (
            software_path, setting_path, network_path)

        data = []
        for i in range(num):
            cache_dir = '%s/cache/%s' % (self.dream_path, self.name)
            check_dir_and_make(cache_dir)
            subprocess.check_call(cmd, shell=True, cwd=cache_dir, stderr=subprocess.DEVNULL)
            data.append(pd.read_csv('%s/%s_wildtype.tsv' % (cache_dir, self.name), sep='\t'))
            shutil.rmtree(cache_dir)
        data = pd.concat(data).to_numpy()
        return data

    def single_sample(self, j, v):
        software_path = '../../gnw-3.1b.jar'
        setting_path = '../../settings.txt'
        network_path = '../../configurations/%s.xml' % self.name
        cmd = 'java -jar %s -c %s --input-net %s --output-net-format=1 --simulate' % (
            software_path, setting_path, network_path)

        cache_dir = '%s/cache/%s' % (self.dream_path, self.name)
        check_dir_and_make(cache_dir)
        subprocess.check_call(cmd, shell=True, cwd=cache_dir, stderr=subprocess.DEVNULL)
        data = pd.read_csv('%s/%s_knockouts.tsv' % (cache_dir, self.name), sep='\t')
        shutil.rmtree(cache_dir)
        data = data.iloc[j].to_numpy()

        return data

    def batch_sample(self, intervention_list):
        samples = []
        for (j, v) in intervention_list:
            samples.append(self.single_sample(j, v))
        return np.array(samples)

    def pass_forward(self, base_samples):
        raise "DREAM can not express the pass_forward function."
