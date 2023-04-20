import torch
import torch.nn as nn
import torch.nn.functional as F


class nll_loss():
    def __init__(self,**argv):
        self.argv = argv
    
    def __call__(self,input,target):
        return F.nll_loss(torch.log(input),target)


class brier_loss():
    def __init__(self):
        pass
    def __call__(self,input,target):

        for pr,gt in zip(input, target):
            max_idx = torch.argmax(pr)
            max_value = pr[max_idx]

            n_classes = len(pr)

            one_hot = torch.zeros(n_classes)
            one_hot[gt] = 1

            barrier = (max_value - one_hot[max_idx])**2
        
        return barrier

        
