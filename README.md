## Bayesian Active Causal Discovery with Multi-Fidelity Experiments

## 1 Abstract

Previous active causal discovery methods regard the data are all from one fidelity, but not from multi-fidelity. In real-world, however, interventional experiments can always acquire from multi-fidelity, where higher-fidelity experiments are more accurate but expensive, while lower ones are cheaper but noisy. Comparing with single-fidelity settings, the trade-off between information and cost under different fidelities should be emphasized, where we intend to use cheap but sufficiently informative data to reveal the real causal graph. Challenges come from the fidelity selection process under the trade-off, as well as the optimization process under various noise from different fidelities. In this paper, we formally define the problem of multi-fidelity active causal discovery, and design a Bayesian framework, called \textit{Licence}, to solve this problem. We propose a probabilistic cascading model to capture the correlations between different fidelities, and design a mutual-information based acquisition function to conduct intervention and fidelity selection. Moreover, we extend the single intervention to batch scenario. We find the theoretical foundations behind greedy method do not hold in our problem. For alleviating this issue, we introduce a new concept called $\epsilon$-submodular, and design a constraint-based fidelity model to bound the optimal solution from greedy strategy. We conduct extensive experiments to demonstrate the effectiveness of our model, and release our project at this site.

## 2 Contributions

In conclusion, the main contributions of this paper can be summarized as follows:

- We formally define the task of active causal discovery with multi-fidelity oracles, which, to our knowledge, is the first time in the field of causal discovery.

- We propose a Bayesian framework called \textit{Licence}, which is composed of a probabilistic cascaded fidelity model and a mutual information based acquisition function, in order to solve the task.

- We extend our framework to the batch intervention scenario. We define $\epsilon$-submodular, and develop a constraint based fidelity model, which provides theoretical guarantees for the efficient greedy method.

- We conduct extensive experiments to demonstrate the effectiveness of our model. In order to promote this research direction, we have released our project at Github.

## 4 Quick Start

### Step 1: Download the project

First of all, download our project `Licence.zip` from [Github](https://github.com/anonymous4sup/Licence/tree/main/project) and unzip the file. The file includes both codes and datasets.

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
