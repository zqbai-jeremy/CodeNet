# CodeNet

This is the Python implementation of MDS based strategy in CodeNet with soft error simulation, writen by [Ziqian Bai (Jeremy)](https://github.com/zqbai-jeremy).

## Prerequisites
- A usable Amazon EC2 account
- Python 2.7 with StarCluster toolkit

## Setup
- Follow the [tutorial](http://mpitutorial.com/tutorials/launching-an-amazon-ec2-mpi-cluster/) to setup the cluster. For the experiments in the paper, 40 m3.medium instances are used.
- Clone the repo to /home/\<username\>
```bash
git clone https://github.com/zqbai-jeremy/CodeNet.git
cd CodeNet
```

## Usage
```bash
mpiexec -n <# of processors> python codedDNN_CNN.py <strategy type> <network type> <checkpoint freq> <mode> <train set size> <test set size> <round>
```
- **\# of processors**: total number of processors used [40 in the paper]
- **strategy type**: what strategy to use; choose from mds/replica/uncoded
- **network type**: what network to use; currently only "fc"(fully connected) is available
- **checkpoint freq**: the number of iterations to do check-pointing
- **mode**: which mode to use; choose from -time/-accuracy; -time: record the time of every iteration and only test accuracy at the end; -accuracy: test the accuracy at every round
- **train set size**: number of data instances used for training [2000 in the paper]
- **test set size**: number of data instances used for testing [500 in the paper]
- **round**: split the training process to how many rounds (train "train set size"/"round" data instances and test the accuracy at each round); only for -accuracy mode [10 in the paper]
