# M/V-ATTA


## Introduction

This page presents the code for the uncertainty calibration methods M-ATTA and V-ATTA, presented in the paper "Approaching Test Time Augmentation in the Context of Uncertainty Calibration for Deep Neural Networks", submitted to the IEEE Transactions on Pattern Analysis and Machine Intelligence. A pre-print version of this paper can be found at: https://arxiv.org/abs/2304.05104

![model_highlevel_2](https://github.com/pedrormconde/MV-ATTA/assets/74596251/098bdfa0-a568-4ba5-b342-2fb665873255)



## Instructions

We refer to Secion 4 of the referred paper, before using the code here presented, in order for the user to understand how to construct th inputs for our models. 

The inputs for training and testing to be used with our models must be located in the directories 'data/train' and 'data/test', respectively. The inputs must be of the form:

- 'Z.npy', a numpy array of shape $(N,m,k)$ where $N$ equals number of samples in the set, $m$ equal the number of different augmentations used, and $k$ the number of classes in the problem. For each sample, the array 'Z.npy' represents the transpose of the matrix $\mathbf{Z} \in \mathbb{R}^{k,m}$ introduced in the article.

- 'p0.npy', a numpy array of shape $(N,k)$ where $N$ equals number of samples in the set and $k$ the number of classes in the problem. For each sample, the array 'p0.npy' represents the vector $\mathbf{p}_0 \in \mathbb{R}^{k}$ introduced in the article.

- 'labels.npy', a numpy array of shape $(N,)$ where $N$ equals number of samples in the set. Each sample in the array 'labels.npy' is the true class of the respective sample in the previous arrays.

To train and test both M-ATTA and V-ATTA we refer to the files 'M-ATTA.py' and 'V-ATTA.py', respectively. In both files we find

```
if __name__=='__main__':
    
    TRAIN(epochs=500, learning_rate=0.001, batch_size=500)
    
    TEST(checkpoint_file='checkpoint.pth')
```

In the 'TRAIN' function you can edit the number of epochs, the learning rate and the batch size. This function will optimize the parameters of the model using the data present in 'data/train' and the Negative Log-Likelihood (NLL) as loss function. This function will print the results for that same set of data, with respect to the Brier score, the Expected Calibration Error (ECE), the multi-class Brier score and the NLL. Also, it will produce a file with the optimized parameters of the model in 'checkpoints/checkpoint.pth'.

In the 'TEST' function you can edit the name of the file from where you load the parameters of the model that will be tested. The file is expected to be in the directory 'checkpoints'. This function will print the results for the set of data present in 'data/test', with respect to the same metrics as before. Also, it will produce a file with the final prediction probability vector, for all samples, in the form of an array with shape $(N,k)$ (where $N$ in the number of samples in the test set, and $k$ the number of classes in the problem) saved as 'prediction/prediction.npy'.
