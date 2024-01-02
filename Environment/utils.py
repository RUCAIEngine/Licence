import Environment.Dataset as Dataset


def create_causal_graph(config):
    if config['dataset'] == 'ERGraph':
        return Dataset.ERGraph(config)

    if config['dataset'] == 'SFGraph':
        return Dataset.SFGraph(config)

    if config['dataset'][:5] == 'DREAM':
        return Dataset.DreamGraph(config)

    raise "Error in loading dataset!"
