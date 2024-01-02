## Bayesian Active Causal Discovery with Multi-Fidelity Experiments

## 1 Abstract

This paper studies the problem of active causal discovery when the experiments can be done based on multi-fidelity oracles, where higher fidelity experiments are more precise and expensive, while the lower ones are cheaper but less accurate. In this paper, we formally define the task of multi-fidelity active causal discovery, and design a probabilistic model for solving this problem. In specific, we first introduce a mutual-information based acquisition function to determine which variable should be intervened at which fidelity, and then a cascading model is proposed to capture the correlations between different fidelity oracles. Beyond the above basic framework, we also extend it to the batch intervention scenario. We find that the theoretical foundations behind the widely used and efficient greedy method do not hold in our problem. To solve this problem, we introduce a new concept called $\epsilon$-submodular, and design a constraint based fidelity model to theoretically validate the greedy method. We conduct extensive experiments to demonstrate the effectiveness of our model.

## 2 Contributions

In conclusion, the main contributions of this paper can be summarized as follows:

- We formally define the task of active causal discovery with multi-fidelity oracles, which, to our knowledge, is the first time in the field of causal discovery.

- To solve the above task, we propose a Bayesian framework, which is composed of a mutual information based acquisition function and a cascading fidelity model.

- To extend our framework to the batch intervention scenario, we introduce a constraint-based fidelity model, which provides theoretical guarantees for the efficient greedy method.

- We conduct extensive experiments to demonstrate the effectiveness of our model. In order to promote this research direction, we have released our project at Github.

## 3 Quick Start

### Step 1: Download the project

First of all, download our project from Github. These files include both codes and datasets.

### Step 2: Create the running environment

Create `Python` enviroment and install the packages that the project requires.
- networkx
- graphical_models
- igraph
- bayesian-optimization
- cdt

You can install the packages with the following command.

```
    pip install -r requirements.txt
```

### Step 3: Run the project

Choose a dataset to run (e.g. ERGraph) to run with the following command.

```
    python run.py ERGraph 10 Licence Licence single 
```

You can also change the hyper-parameters as you want.

## Citation

If you find this paper useful, please cite our paper:

```
@inproceedings{zhang2023bayesian,
  title={Bayesian Active Causal Discovery with Multi-Fidelity Experiments},
  author={Zhang, Zeyu and Li, Chaozhuo and Chen, Xu and Xie, Xing},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```
