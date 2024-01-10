# Loss of Plasticity in Deep Continual Learning
This repository contains the implementation of three continual supervised learning problems.
In our forthcoming paper _Maintaining Plasticity in Deep Continual Learning_, 
we show the loss of plasticity in deep learning in these problems.

A talk about this work can be found [here](https://www.youtube.com/watch?v=p_zknyfV9fY), 
and the [paper](https://arxiv.org/abs/2306.13812) is available on arxiv.

# Run on Euler cluster
First change to the following module

```sh
module load gcc/6.3.0 python/3.8.5
```

# Installation

```sh
virtualenv --python=/usr/bin/python3.8 loss-of-plasticity/
source loss-of-plasticity/bin/activate
cd loss-of-plasticity
pip3 install -r requirements.txt
pip3 install -e .
```

Add these lines in your .zshrc
```sh
source PATH_TO_DIR/loss-of-plasticity/lop/bin/activate
export PYTHONPATH=$PATH:PATH_TO_DIR/lop 
```
