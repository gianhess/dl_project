# Can We Understand Plasticity Through Neural Collapse?
This is a github repository for a project for Deep Learning course at ETH Zurich in the HS23. In this research project we embarked on a journey to discover and analyze to connection between plasticity loss and neural collapse.

Project members:
* Guglielmo Bonifazi
* Iason Chalas 
* Gian Hess
* Jakub Łucki 

## Experiments
In the project proposal we decided to perform three experiments:

* `experiments/warm_up/` - Warm Up experiment. First training on a half of the dataset then training on the full dataset.
    * `run.py` - main script for running experiments
    * `results*/` - folders for storing the results
* `experiments/continual_experiments/lop/permuted_mnist/` - Permuter MNIST experiment. A sequence of tasks consisted of permuted images
    * `cfg/` - configurations of different versions of the experiment
    * `analysis/` - notebooks used to plot and analyse the results.
    * `load_mnist.py` - loads the dataset
    *  `multi_param_expr.py` - creates helper configuration files
    * `online_expr*.py` - scripts for runnning the experiment
* `experiments/continual_experiments/lop/imagenet/` - Continual ImageNet experiment not used in the report, first pilot experiments have shown that it is too computationally expensive

## Neural collapse
In the folder `util/` we have a script `neural_collapse.py` which is used to compute the 4 metrics of neural collapse.