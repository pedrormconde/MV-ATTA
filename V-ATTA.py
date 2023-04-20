
import numpy as np
from utils.utils import *
import os

from model.netcalib import M_ATTA, V_ATTA

from calibration import Calibration, to_format_dict


data_dir = 'data'


P = np.load(os.path.join(data_dir,'train','Z.npy'))

n_classes = P.shape[2]
n_augms = P.shape[1]


def load_data(root,dataset):
    p0_file = os.path.join(root,dataset,'p0.npy')
    assert os.path.isfile(p0_file)
    p0_data = np.load(p0_file)
    
    p_file = os.path.join(root,dataset,'Z.npy')
    assert os.path.isfile(p_file)
    p_data = np.load(p_file)

    pgt_file = os.path.join(root,dataset,'labels.npy')
    assert os.path.isfile(pgt_file)
    pgt_data = np.load(pgt_file).astype(np.int64)

    # data has to have the same number of samples
    assert p0_data.shape[0] == p_data.shape[0]
    assert p0_data.shape[0] == pgt_data.shape[0]
    
    p_data =np.swapaxes(p_data,axis1=2,axis2=1)
    samples = p0_data.shape[0]
    p0_data = np.reshape(p0_data,(samples,n_classes,1))

    return {'p0':p0_data,'labels':pgt_data,'p':p_data}


def TRAIN(epochs, learning_rate, batch_size):
   
    
    training_set = load_data(data_dir,'train')


    # Define calibration model
    model = M_ATTA(n_classes,n_augms)
    calib = Calibration(model)
    
    calib.Train( training_set, training_set, loss = 'nll_loss', epochs = epochs, lr = learning_rate, batch_size = batch_size, test_priod = epochs, plot=False)
       
    

def TEST(checkpoint_file):


    test_set     = load_data(data_dir,'test')


    # Define calibration model
    model = M_ATTA(n_classes,n_augms)
    
       
    calib = Calibration(model, checkpoints=checkpoint_file)
    
    
    test_set  = to_format_dict(test_set)
    prediction,_ = calib._test_epoch(0,test_set)
    np.save(os.path.join('prediction','P'), prediction)



if __name__=='__main__':
    
    TRAIN(epochs=500, learning_rate=0.001, batch_size=500)
    
    TEST(checkpoint_file='checkpoint.pth')
        