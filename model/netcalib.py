import torch
import torch.nn as nn
import torch.nn.functional as F



    
    
class M_ATTA(nn.Module):
    def __init__(self,n_classes,n_augms) -> None:
        super(M_ATTA,self).__init__()
        self.n_classes = n_classes
        self.n_augms = n_augms
        
        self.W      = nn.Parameter((1/n_classes) * torch.ones((n_classes,n_augms)))
           
        self.norm   = nn.Softmax(dim=0)
        self.omega  = nn.Parameter(torch.ones(1))
        #self.op     = nn.Parameter(torch.ones([self.n_classes,1]))
        self.op     = torch.ones([n_augms,1])


    def forward(self,p0,P):
        omega_ = torch.clamp(self.omega,min=0,max=1)

        p0_c = torch.argmax(p0,dim=0)
        #max_range = omega_.detach().cpu().item()*100
        #print(max_range)
        #assert max_range >= 1, "Omega is < 1 "
        for i in range(int(1001)):
            #if self.training:
            #    rand_w = torch.randn(self.W.shape) * 0.2
            #else:
            aug_W = self.W 
            #print(self.W.shape)
            #print(P.shape)
            wp = torch.mul(aug_W,P)
            #print(wp.shape)
            pp = torch.matmul(wp,self.op)
            WP_nom = self.norm(pp)

            # Ensure that WP_nom is normalized
            #if not abs(torch.sum(WP_nom,dim=0).detach().numpy().item() - 1.0) < 0.01:
                #print(torch.sum(WP_nom,dim=0).detach().numpy().item())
                #print(aug_W)
            #assert  abs(torch.sum(WP_nom,dim=0).detach().numpy().item() - 1.0) < 0.01
            
            p = (1-omega_)*p0 + omega_* WP_nom
            # Ensure that p is normalized
            #if not abs(torch.sum(p,dim=0).detach().numpy().item() - 1.0) < 0.01:
                #print(torch.sum(p,dim=0).detach().numpy().item())
            #assert  abs(torch.sum(p,dim=0).detach().numpy().item() - 1.0) < 0.01

            p_c = torch.argmax(p,dim=0)
        
            if p_c == p0_c:
                break
            omega_ = omega_ - 0.01 
            if omega_<0.001: 
                omega_=0
                break
        return(torch.swapaxes(p,axis0=1,axis1=0))
    
    
class V_ATTA(nn.Module):
    def __init__(self,n_classes,n_augms) -> None:
        super(V_ATTA,self).__init__()
        self.n_classes = n_classes
        self.n_augms = n_augms
        
        self.W      = nn.Parameter((1/n_classes) * torch.ones((1,n_augms)))
        
        self.norm   = nn.Softmax(dim=0)
        self.omega  = nn.Parameter(torch.ones(1))
        #self.op     = nn.Parameter(torch.ones([self.n_classes,1]))
        self.op     = torch.ones([n_augms,1])


    def forward(self,p0,P):
        omega_ = torch.clamp(self.omega,min=0,max=1)

        p0_c = torch.argmax(p0,dim=0)
        #max_range = omega_.detach().cpu().item()*100
        #print(max_range)
        #assert max_range >= 1, "Omega is < 1 "
        for i in range(int(1001)):
            #if self.training:
            #    rand_w = torch.randn(self.W.shape) * 0.2
            #else:
            aug_W = self.W 
            #print(self.W.shape)
            #print(P.shape)
            wp = torch.mul(aug_W,P)
            #print(wp.shape)
            pp = torch.matmul(wp,self.op)
            WP_nom = self.norm(pp)

            # Ensure that WP_nom is normalized
            #if not abs(torch.sum(WP_nom,dim=0).detach().numpy().item() - 1.0) < 0.01:
                #print(torch.sum(WP_nom,dim=0).detach().numpy().item())
                #print(aug_W)
            #assert  abs(torch.sum(WP_nom,dim=0).detach().numpy().item() - 1.0) < 0.01
            
            p = (1-omega_)*p0 + omega_* WP_nom
            # Ensure that p is normalized
            #if not abs(torch.sum(p,dim=0).detach().numpy().item() - 1.0) < 0.01:
                #print(torch.sum(p,dim=0).detach().numpy().item())
            #assert  abs(torch.sum(p,dim=0).detach().numpy().item() - 1.0) < 0.01

            p_c = torch.argmax(p,dim=0)
        
            if p_c == p0_c:
                break
            omega_ = omega_ - 0.01 
            if omega_<0.001: 
                omega_=0
                break
        return(torch.swapaxes(p,axis0=1,axis1=0))