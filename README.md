# M/V-ATTA

(Under construction)

## Introduction

This page presents the code for the uncertainty calibration methods M-ATTA and V-ATTA, presented in the paper "Approaching Test Time Augmentation in the Context of Uncertainty Calibration for Deep Neural Networks", submitted to the IEEE Transactions on Pattern Analysis and Machine Intelligence. A pre-print version of this paper can be found at: https://arxiv.org/abs/2304.05104


## Instructions

We refer to Secion 4 of the referred paper, before using the code here presented. 

The inputs for training and testing to be used with our models must be located in the directories 'data/train' and 'data/test', respectively. The inputs must be of the form:

- 'Z.npy', a numpy array of shape $(N,m,k)$ where $N$ equals number of samples in the set, $m$ equal the number of different augmentations used, and $k$ the number of classes in the problem. For each sample, the array 'Z.npy' represents the transpose of the matrix $\mathbf{Z}$ introduced in the article.

- 'p0.npy', a numpy array of shape $(N,k)$ where $N$ equals number of samples in the set and $k$ the number of classes in the problem. For each sample, the array 'p0.npy' represents the vector $\mathbf{p}_0$ introduced in the article.

- 'lables.npy', a numpy array of shape $(N,)$ where $N$ equals number of samples in the set. Each sample in the array 'lables.npy' is the true class of the respective sample in the previous arrays.
