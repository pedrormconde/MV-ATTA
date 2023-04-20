import torch
import torch.nn as nn
import numpy as np

import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import uncertainty_metrics.numpy as um
import torch.nn.functional as F


def gen_data(n_classes,n_aug, verbose=False):
    size = (n_classes,n_aug)
    
    norm = nn.Softmax(dim=0)
    p0 = norm(torch.tensor(np.random.random(size=(n_classes,1))))
    #p0 = norm(p0)
    p = norm(torch.tensor(np.random.random(size=(n_classes,n_aug))))
    p_gt = torch.argmax(norm(torch.tensor(np.random.random(size=(1,n_classes)))),dim=1)
    #p_gt = torch.cat([p_gt]*2,dim=0)

    if verbose:
        print('*'*50)
        print('P_zero = \n')
        print(p0.numpy())
        print(torch.sum(p0,dim=0).numpy())

        print('*'*50)
        print('P = \n')
        print(p.numpy())
        print(torch.sum(p,dim=0).numpy())


        print('*'*50)
        print('P_gt = \n')
        print(p_gt.numpy())

    return {'p0':p0,'p':p,'p_gt':p_gt}



def to_format(in_v):
    if not torch.is_tensor(in_v):
        in_v = torch.tensor(in_v)
    in_v = in_v.type(torch.float32)
    return in_v


def kl_divergence(p, q):
    kl = p * torch.log2(p / q)
    out = torch.nansum(kl)
    return out


def plot_disto(pgt_data,p0_data,prediction,axis):
    p0_value = []
    p_value  = []
    
    for i,pgt in enumerate(pgt_data):
        p0_value.append(p0_data[i,pgt].item())
        p_value.append(prediction[i,pgt].item())    

    datf = pd.DataFrame({"calibrated": p_value,"uncalibrated" : p0_value})
    sns.histplot(data=datf,bins = 50,kde=True, ax=axis,fill=True,element="step")
    sns.move_legend(axis, "upper left")




def brier_score(pred_in, gt_in):

    if  torch.is_tensor(pred_in):
        pred_in = pred_in.detach().numpy()
    
    if  torch.is_tensor(gt_in):
        #gt_in = gt_in.detach().numpy().astype(np.int32).squeeze()
        gt_in = gt_in.detach().numpy().astype(np.int32)

    brier_array = []
    for pr,gt in zip(pred_in, gt_in):
        max_idx = np.argmax(pr)
        max_value = np.max(pr)

        n_classes = len(pr)

        one_hot = np.zeros(n_classes)
        one_hot[gt] = 1

        brier_array.append((max_value - one_hot[max_idx])**2)

    return(np.mean(brier_array))




def mc_brier(pred_in, gt_in):

    if  torch.is_tensor(pred_in):
        pred_in = pred_in.detach().numpy()
    
    if  torch.is_tensor(gt_in):
        #gt_in = gt_in.detach().numpy().astype(np.int32).squeeze()
        gt_in = gt_in.detach().numpy().astype(np.int32)

    brier_array = []
    for pr,gt in zip(pred_in, gt_in):
        
        n_classes = len(pr)
        one_hot = np.zeros(n_classes)
        one_hot[gt] = 1
        for i in range(len(pr)):
            brier_array.append((pr[i] - one_hot[i])**2)

    return(sum(brier_array)/len(gt_in))





def ece(pred_in, gt_in, List_bins=[15]):
    
    if  torch.is_tensor(pred_in):
        pred_in = pred_in.detach().numpy()
    
    if  torch.is_tensor(gt_in):
        #gt_in = gt_in.detach().numpy().astype(np.int32).squeeze()
        gt_in = gt_in.detach().numpy().astype(np.int32)
    
    preds = []
    labels_onehot = []
    
    for pr,gt in zip(pred_in, gt_in):
        max_idx = np.argmax(pr)
        max_value = pr[max_idx].item()

        n_classes = len(pr)

        one_hot = np.zeros(n_classes)
        one_hot[gt] = 1
        
        label_onehot = one_hot[max_idx]
        
        preds.append(max_value)
        labels_onehot.append(label_onehot)
        
        
    #REMOVING ELEMENTS##########################################
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    edges = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    remove = []
    
    for i in range(len(preds)):
        for edge in edges:
            if abs(preds[i]-edge)<0.001:
                remove.append(i)
    
    preds = np.array(preds).flatten()
    labels_onehot = np.array(labels_onehot, dtype=int).flatten()
                
    preds = np.delete(preds,remove)
    labels_onehot = np.delete(labels_onehot,remove)
    
    print(len(remove))
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #############################################################

    
    preds = np.array(preds).flatten()
    labels_onehot = np.array(labels_onehot, dtype=int).flatten()
    
    
    
    List_ECE=[]
    
    for num_bins in List_bins:
    
        bins = np.linspace(0.0, 1.0, num_bins+1)#[1:]
        bins[0] = bins[0]-0.001
        binned = np.digitize(preds, bins, right=True)-1
            
        bin_accs = np.zeros(num_bins)
        bin_confs = np.zeros(num_bins)
        bin_sizes = np.zeros(num_bins)
        
        for i in range(num_bins):
            bin_sizes[i] = len(preds[binned == i])
            if bin_sizes[i] > 0:
              bin_accs[i] = (labels_onehot[binned==i]).sum() / bin_sizes[i]
              bin_confs[i] = (preds[binned==i]).sum() / bin_sizes[i]
        
        ECE = 0
        #MCE = 0
        
        #size_bin = []
        #calib_dif_bin = []
      
        for i in range(num_bins):
          abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
          ECE += (bin_sizes[i] ) * abs_conf_dif
          #print(f'{i} : {abs_conf_dif}, {bin_sizes[i]}')
          #size_bin.append(bin_sizes[i])
          #calib_dif_bin.append(abs_conf_dif)
          #MCE = max(MCE, abs_conf_dif)
        
        #return ECE, MCE, size_bin, calib_dif_bin
        
        #ECE = um.ece(labels_onehot, preds, num_bins=num_bins)
        ECE = ECE/sum(bin_sizes)
    
        List_ECE.append(ECE)
    
    return sum(List_ECE) / len(List_ECE)
    




def t_ece(pred_in, gt_in, List_bins=[15], thresh=0.05):
    
    if  torch.is_tensor(pred_in):
        pred_in = pred_in.detach().numpy()
    
    if  torch.is_tensor(gt_in):
        #gt_in = gt_in.detach().numpy().astype(np.int32).squeeze()
        gt_in = gt_in.detach().numpy().astype(np.int32)
    
    preds = []
    labels_onehot = []
    
    for pr,gt in zip(pred_in, gt_in):
            
        n_classes = len(pr)
        one_hot = np.zeros(n_classes)
        one_hot[gt] = 1
    
        for i in range(len(pr)):
            if pr[i]>thresh:
                preds.append(pr[i])
                labels_onehot.append(one_hot[i])
        

    
    preds = np.array(preds).flatten()
    labels_onehot = np.array(labels_onehot, dtype=int).flatten()
    
    
    
    List_ECE=[]
    
    for num_bins in List_bins:
    
        bins = np.linspace(0.0, 1.0, num_bins+1)#[1:]
        bins[0] = bins[0]-0.001
        binned = np.digitize(preds, bins, right=True)-1
            
        bin_accs = np.zeros(num_bins)
        bin_confs = np.zeros(num_bins)
        bin_sizes = np.zeros(num_bins)
        
        for i in range(num_bins):
            bin_sizes[i] = len(preds[binned == i])
            if bin_sizes[i] > 0:
              bin_accs[i] = (labels_onehot[binned==i]).sum() / bin_sizes[i]
              bin_confs[i] = (preds[binned==i]).sum() / bin_sizes[i]
        
        ECE = 0

      
        for i in range(num_bins):
          abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
          ECE += (bin_sizes[i] ) * abs_conf_dif

        ECE = ECE/sum(bin_sizes)
    
        List_ECE.append(ECE)
    
    return sum(List_ECE) / len(List_ECE)



def neg_ll(pred_in, gt_in,):
    
    gt_in = np.squeeze(gt_in)
    nll_array = []
    
    if not torch.is_tensor(pred_in):
        pred_in = torch.from_numpy(pred_in)
    
    if not torch.is_tensor(gt_in):
        #gt_in = gt_in.detach().numpy().astype(np.int32).squeeze()
        gt_in = torch.from_numpy(gt_in)
    
    gt_in = gt_in.type(torch.LongTensor)
    
    """
    for pr,gt in zip(pred_in, gt_in):
        pr = pr.type(torch.LongTensor)
        gt = gt.type(torch.LongTensor)
        pr = torch.unsqueeze(pr,0)
        nll_array.append(F.nll_loss(torch.log(pr),gt))
    """
       
    return(F.nll_loss(torch.log(pred_in),gt_in))




def acc(pred_in, gt_in):
    
    if  torch.is_tensor(pred_in):
        pred_in = pred_in.detach().numpy()
    
    if  torch.is_tensor(gt_in):
        #gt_in = gt_in.detach().numpy().astype(np.int32).squeeze()
        gt_in = gt_in.detach().numpy().astype(np.int32)

    total=0
    count=0 
    
    for pr,gt in zip(pred_in, gt_in):
        total=total+1
        max_idx = np.argmax(pr)
        
        if max_idx == gt:
            count=count+1

    return (count/total)*100