import argparse
import csv
import random
import os
from copy import copy

import networkx as nx
import numpy as np
import torch
from Environment import Environment
import Model as models
import cdt
from utils import *
import config as config_files

avg_num = 10


def run(config, mmetrics):
    Model = eval('models.%s' % config['model'])

    environment = Environment(config)
    model = Model(config)

    # Training on observational data
    observational_data = environment.initial_observational_samples()
    model.observational_train(observational_data)
    print('Finish observational training.')

    # Active learning
    if config['acq_param']['acq_type'] == 'single':
        consumption = 0
        step = 0
        while consumption + config['fidelity_cost_list'][0] <= config['acq_param']['acq_budge']:
            # Generate single query
            single_query = model.generate_single_query()
            # If not consume the m from alg, try the best fidelity as it could.
            if consumption + config['fidelity_cost_list'][single_query[2]] > config['acq_param']['acq_budge']:
                m_best = find_best_fidelity(config['acq_param']['acq_budge'] - consumption,
                                            config['fidelity_cost_list'])
                single_query = (single_query[0], single_query[1], m_best)
            consumption += config['fidelity_cost_list'][single_query[2]]

            # Acquire interventional data
            single_interventional_data = environment.single_acquire(single_query)

            # Update model
            model.single_update(single_query, single_interventional_data)
            step += 1

            print('---Step[%d]---' % step)
            print(single_query)

    elif config['acq_param']['acq_type'] == 'batch':
        for step in range(config['acq_param']['acq_max_step']):
            # Generate batch query
            batch_query = model.generate_batch_query()

            # Acquire interventional data
            batch_interventional_data = environment.batch_acquire(batch_query)

            # Update model
            model.batch_update(batch_query, batch_interventional_data)
            print('---Step[%d]---' % step)
            print(batch_query)

    collector = MetricCollector(mmetrics)
    if config['details']:
        detail_list = []

    # Evaluation
    for i in range(avg_num):

        graph = model.output_graph()
        target = environment.true_causal_graph

        graph_adj = graph.get_topological_graph()
        target_adj = nx.to_numpy_array(target.get_topological_graph(), dtype=int)

        # 1 SHD (topological property)
        SHD = cdt.metrics.SHD(target_adj, graph_adj)
        # 2 AUC_PR (topological property)
        AUC_PR, PRC = cdt.metrics.precision_recall(target_adj, graph_adj)
        # 3 RMSE (functional property)
        if config['dataset'] != 'DREAM_E' and config['dataset'] != 'DREAM_Y':
            RMSE = graph_rmse(target, graph)
            collector.add_all({'SHD': SHD, 'AUPRC': AUC_PR, 'RMSE': RMSE})
        else:
            collector.add_all({'SHD': SHD, 'AUPRC': AUC_PR})

        # Detect graph details.
        if config['details']:
            graph_dict = {'true_graph': target_adj, 'SHD': SHD}
            fidelity_graph_list = []
            for m in range(len(config['fidelity_noise_list'])):
                g_tmp = model.output_graph_details(m)
                fidelity_graph_list.append(g_tmp)
            graph_dict['discovery_graph'] = fidelity_graph_list
            detail_list.append(graph_dict)
    if config['details']:
        path = 'graph_%s' % get_local_time()
        write_pickle(detail_list, path)

    performance = collector.output_avg()

    return performance


def avg_running(config):
    if config['dataset'] != 'DREAM_E' and config['dataset'] != 'DREAM_Y':
        mmetrics = ['SHD', 'AUPRC', 'RMSE']
    else:
        mmetrics = ['SHD', 'AUPRC']

    meta_collector = MetaMetricCollector(mmetrics)
    if config['debug'] == 0:
        i = 0
        while i < avg_num:
            try:
                performance = run(config, mmetrics)
                print('Repeat[%d]:' % i, performance)
                meta_collector.add_performance(performance)
                i += 1
            except Exception as e:
                print(e)
                continue
    else:
        i = 0
        while i < avg_num:
            performance = run(config, mmetrics)
            print('Repeat[%d]:' % i, performance)
            meta_collector.add_performance(performance)
            i += 1
    meta_mean, meta_var = meta_collector.output_avg()
    tag = '%s_%s_%d_%s_%s' % (config['acq_param']['acq_type'], config['dataset'], config['node_num'], config['model'],
                              config['fidelity_strategy'])
    if config['acq_param']['acq_type'] == 'single':
        budge = '%d' % config['acq_param']['acq_budge']
    else:
        budge = '%d' % config['acq_param']['acq_batch_budge']
    data = [get_local_time(), tag, budge]
    for k in mmetrics:
        data.append(meta_mean[k][0])
        data.append(meta_mean[k][1])

    if config['output_path'] == 'default':
        check_dir_and_make('result')
        output_path = 'result/%s_%s_%d_%s_%s' % (
        config['acq_param']['acq_type'], config['dataset'], config['node_num'], config['model'],
        config['fidelity_strategy'])
    else:
        output_path = config['output_path']

    with open(output_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows([data])


def convert_hyper_param(extra_args):
    """
    Convert string hyper-parameters into dicts.
    :param extra_args: (str)  hyper-parameters with string form.
    :return: (dict) the dict of hyper-parameters
    """
    hyperParam = {}
    tmpKey, tmpValue = None, None
    for index, ext in enumerate(extra_args):
        if index % 2 == 0:
            tmpKey = ext
        else:
            if ext.isdigit():
                tmpValue = int(ext)
            else:
                tmpValue = float(ext)
            hyperParam[tmpKey] = tmpValue
    return hyperParam


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('node_num', type=int)
    parser.add_argument('model', type=str)
    parser.add_argument('fidelity_strategy', type=str)
    parser.add_argument('acquisition_type', type=str)
    parser.add_argument('-budge', default=-1, type=int)

    parser.add_argument('-seed', default=1128, type=int)
    parser.add_argument('-cuda', default=0, type=int)
    parser.add_argument('-debug', default=0, type=int)
    parser.add_argument('-details', default=0, type=int)

    args, extra_args = parser.parse_known_args()

    set_seed(args.seed)
    if args.cuda != 0:
        torch.cuda.set_device(args.cuda)

    config = copy(config_files.common_config)
    config['dataset'] = args.dataset
    config['node_num'] = args.node_num
    config['model'] = args.model
    config['fidelity_strategy'] = args.fidelity_strategy

    config['dataset_param'] = eval('config_files.%s_config' % args.dataset)
    config['model_param'] = eval('config_files.%s_config' % args.model)
    config['acq_param'] = eval('config_files.%s_config' % args.acquisition_type)
    config['debug'] = args.debug
    config['details'] = args.details
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    config['device'] = device

    hyper_param = convert_hyper_param(extra_args)
    for name, param in hyper_param.items():
        config['model_param'][name] = param

    print(config['model_param'])

    if args.budge != -1:
        if args.acquisition_type == 'single':
            config['acq_param']['acq_budge'] = args.budge
        else:
            config['acq_param']['acq_batch_budge'] = args.budge

    print('Environment: %s_%d_%s' % (
    config['dataset'], config['node_num'], config['fidelity_strategy']))
    print('Model: %s' % config['model'])
    print('seed[%d] cuda[%d]' % (args.seed, args.cuda))

    avg_running(config)
    print('Running Finish!')
