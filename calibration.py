import numpy as np
#from utils import *
from utils.utils import *
from utils.loss import *
import utils.loss as loss_lib
from tqdm import tqdm
import os


def to_format_dict(data):
    
    samples = len(data['labels'])
    
    data['labels'] = np.resize(data['labels'],(samples,1))
    data['labels'] = to_format(data['labels'])
    data['p']      = to_format(data['p'])
    data['p0']     = to_format(data['p0'])

    return data

def shuffle_dataset(data):
    train_samples = data['labels'].shape[0]
    idx = torch.randperm(train_samples) # shuffle the dataset at each epoch

    data['labels'] = data['labels'][idx].clone()
    data['p0'] = data['p0'][idx].clone()
    data['p'] = data['p'][idx].clone()


    return data


def plot_data(pgt_data,p0_data,prediction,loss,parameters,axis,name):
    plot_disto(pgt_data,p0_data,prediction,axis[0])
    #self.axis[0].legend(loc='upper left')

    axis[1].plot(loss,c='k',label='Loss')
    axis[1].legend(loc =  "upper left")
    
    data = pd.DataFrame({'W (model)':np.array(parameters['W']).flatten()})
    sns.histplot(data=data,bins = 10,kde=True, ax=axis[2],fill=True,element="step")
    axis[2].set_xlim(xmin=-1,xmax=1)
    sns.move_legend(axis[2], "upper left")

    values = np.array(parameters['omega']).flatten()
    axis[3].plot(values,c='k',label = 'Omega (model)')
    axis[3].legend(loc =  "upper left")

    plt.savefig(name,dpi=100)

    axis[0].cla()
    axis[1].cla() 
    axis[2].cla()
    axis[3].cla()


class Calibration():
    def __init__(self, model,checkpoints=None):
        self.model = model
        
        # Save dir
        self.checkpoint_dir = 'checkpoints'
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
    
        if checkpoints != None:
            resume_path = os.path.join(self.checkpoint_dir,checkpoints + '.pth')
            if os.path.isfile(resume_path):
                self._resume_checkpoint(resume_path)
                print("[INF] Loading checkpoints: " + resume_path)

        self.plot_dir = 'prediction'
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        self.logged_param = {}



    def Train(self,trainingset,testset, loss = 'nll_loss', epochs=100, lr = 0.001, batch_size=1, test_priod = 10, plot = False): 
        '''
        
        
        '''
        # Verify if loss exist
        assert loss in  loss_lib.__dict__,f'Loss does not exist ' + loss
        # Load Loss
        self.loss  = loss_lib.__dict__[loss]()

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = lr, weight_decay=0.00001)
        #self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #                                            optimizer=self.optimizer, 
        #                                            mode     =  'min', #['min','max'] (str)
        #                                            factor   =  0.1,
        #                                            patience =  40, # Number of epochs with no improvement after which learning rate will be reduced. 
        #                                            min_lr =  0.000001
        #                                        )
        self.batch_size   = batch_size

        self.test_period  = test_priod
        loss_array  = []
        brier_array = []

        trainingset_troch = to_format_dict(trainingset.copy())
        testset_troch     = to_format_dict(testset.copy())

        if plot:
            self.fig, self.axis = plt.subplots(4, 1)
            self.fig.set_size_inches(18.5, 10.5)
            plt.ion()

        # Training
        for epoch in range(epochs):
            # Shuffle dataset
            trainingset_troch_random = shuffle_dataset(trainingset_troch)
            # Train one epoch
            loss = self._train_epoch(epoch,trainingset_troch_random)
            # Update schedule
            #self.lr_scheduler.step(loss)
            loss_array.append(loss)
        
            # Testing
            if  (epoch+1) % self.test_period == 0:

                print("="*80)
                print("\n")
                # Test Epoch
                prediction, results_array = self._test_epoch(epoch,testset_troch)
                #print('Val:')
                #prediction_val, results_array_val = self._test_epoch(epoch,trainingset_troch)

                #brier_array.append(epoch_brier_score)

                # Plotting
                if plot:
                    parameters = self.log_model_parameters()
                    name = f'{self.plot_dir }/figure{epoch}.png'
                    plot_data(testset['labels'],testset['p0'],prediction,brier_array,parameters,self.axis,name = name)

                #print("Lr %f"%(self.optimizer.param_groups[0]['lr']))
                print("\n")
                print("="*80)
            
                self._save_checkpoint(epoch)

        return({'loss':loss_array,'results_test':results_array, 'prediction': prediction})
    



    def _train_epoch(self,epoch,dataset):
        
        self.model.train()
        epoch_loss_bag = []

        pgt_data = dataset['labels']
        p0_data = dataset['p0']
        p_data = dataset['p']

        n_samples = pgt_data.shape[0]
        #step = n_samples/self.batch_size
        #beta = 0.7

        #norm_distro = (1/10)*torch.ones((10,1))
        tbar = tqdm(range(0,n_samples,self.batch_size), ncols=80)
       
        prediction = []
        epoch_loss = []
        gt = []

        for i in (range(0,n_samples,self.batch_size)):
            
            self.optimizer.zero_grad()
            
            batch_idx_ = np.array(range(i,(i+self.batch_size))) # Generate batch indices
            batch_loss = 0
            loss = torch.tensor(1)
            # Batch Cycle
            for j in batch_idx_:
                # Get data in the correct format for the model: tensor
                
                #rand_w = torch.randn(p0_data.shape[1:])/((epoch*0.1) +1)
    
                pgt = pgt_data[j].type(torch.long)
                p   = p_data[j] #+ rand_w
                p0  = p0_data[j]
                # Compute the Prediction 
                pred = self.model(p0,p)
                
                #loss  = self.loss(pred,pgt) # - (1-beta)*kl_divergence(p,norm_distro)

                loss  = self.loss(pred,pgt) # - (1-beta)*kl_divergence(p,norm_distro)
                loss =  loss/self.batch_size
                loss.backward()
                
                prediction.extend(pred)
                gt.extend(pgt)
                batch_loss += loss.detach().cpu().numpy().item()
                
            epoch_loss.append(batch_loss)
            
            # Compute mean of the gradients
            self.optimizer.step()
            
            prediction_ = torch.stack(prediction)
            gt_ = torch.stack(gt)
            epoch_brier_score = brier_score(prediction_, gt_)
            #brier_score = brier_score(prediction, pgt_data)
            # Update printed information
            #tbar.set_description('T ({}) | barrier {:.4f} Loss {:.4f}'.format(epoch,np.mean(epoch_loss_bag)))
            tbar.set_description('Train ({}) | Brier {:.4f} Loss {:.4f}'.format(epoch,epoch_brier_score,np.mean(epoch_loss)))
            tbar.update()

        return(np.mean(epoch_loss_bag))

    
    def log_model_parameters(self):
        """
        Saves all network parameters to -> logged_param
        
        """
        #local = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if not name in self.logged_param:
                    self.logged_param[name] = [param.data.detach().numpy().copy()]
                else:
                    value = param.data.detach().numpy().copy()
                    self.logged_param[name].append(value)
        return self.logged_param



    def Test(self,pgt_data,p0_data,p_data,plot_disto = False):
        
        samples = pgt_data.shape[0]
        pgt_data = np.resize(pgt_data,(samples,1))
        pgt_data = to_format(pgt_data)
        p_data   = to_format(p_data)
        p0_data  = to_format(p0_data)

        prediction = self._test_epoch(pgt_data,p0_data,p_data)
        brier_score = brier_score(prediction, pgt_data)

        if plot_disto:
            self._plot_disto(self,pgt_data,p0_data,prediction)

        return(prediction)


    def _test_epoch(self,epoch,dataset):
        self.model.eval()
        
        pgt_data = dataset['labels']
        p0_data  = dataset['p0']
        p_data   = dataset['p']

        prediction = []
        gt = []
        n_samples = pgt_data.shape[0]
        tbar = tqdm(range(0,n_samples,1), ncols=80)
        for p0,p in zip(p0_data,p_data):
            # Compute the Prediction 
            p = self.model(p0,p)

            prediction.append(p.squeeze().detach().cpu().numpy())
            #gt.append(gtp.squeeze().detach().cpu().numpy().astype(np.int32).item())
            #if len(prediction)<2:
            #    continue

            #prediction_ = np.stack(prediction)
            #gt_ = np.stack(gt)
            #epoch_brier_score = brier_score(prediction_, gt_)

            tbar.set_description('Results ({}) | Brier {:.4f}'.format(epoch,0))
            tbar.update()
        # output [n_samples x n_clases] <- input []
        prediction = np.stack(prediction,axis=0)
            
        epoch_brier_score = brier_score(prediction, pgt_data)
        epoch_ece = ece(prediction, pgt_data, List_bins=[15])
        #epoch_acc = acc(prediction, pgt_data)
        epoch_mc_brier = mc_brier(prediction, pgt_data)
        #epoch_t_ece = t_ece(prediction, pgt_data)
        epoch_neg_ll = neg_ll(prediction, pgt_data)
        results_array = np.array([epoch_brier_score,epoch_ece,epoch_mc_brier,epoch_neg_ll])
        tbar.set_description('Results ({}) | Brier score: {:.4f}, ECE: {:.4f} , MC-Brier score: {:.4f}, NLL: {:.4f}'.format(epoch,
                                                                                         epoch_brier_score,
                                                                                         epoch_ece,
                                                                                        
                                                                                         epoch_mc_brier,
                                                                                         
                                                                                         epoch_neg_ll))
 
        #print(neg_ll(prediction, pgt_data))
        tbar.update()
        #print('\n Brier {:.4f}\n'.format(epoch_brier_score))
        return prediction,results_array
    
    
    def _save_checkpoint(self, epoch):
        filename = os.path.join(self.checkpoint_dir, f'checkpoint.pth')
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict()
        }
        torch.save(state, filename)

    def _resume_checkpoint(self, resume_path):
        checkpoint = torch.load(resume_path)
        weights_to_load = checkpoint['state_dict']
        self.model.load_state_dict(weights_to_load)
       


    #print(result)



    