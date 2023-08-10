from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
# from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
# from utils.visualization import tsne_eval, confusion_matrix_vis, pca_eval, calculate_cka
from torch.optim import Optimizer
import contextlib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy
from utils.schedulers import CosineSchedule
from utils.metric import accuracy, AverageMeter, Timer


class NormalNN(nn.Module):
    '''
    Normal Neural Network with SGD for classification
    '''
    def __init__(self, learner_config, model) -> None:
        super().__init__()
        self.log = print
        self.config  = learner_config # learner configuration
        self.out_dim = learner_config['out_dim'] # total output dim
        self.model   = model  # load the model
        self.reset_optimizer = True  # if reset optimizer after every round
        self.batch_size = learner_config['batch_size']  # batch size
        self.tasks = learner_config['tasks'] # task list
        self.top_k = learner_config['top_k'] 
        
        # distillation
        self.DTemp = learner_config['temp']
        self.mu = learner_config['mu']
        self.muProx = learner_config['muProx']
        self.muMoon = learner_config['muMoon']
        self.tau = learner_config['tau']
        self.beta = learner_config['beta']
        self.eps = learner_config['eps']
        self.fedmoon = learner_config['fedmoon']

        # supervised criterion
        self.criterion_fn = nn.CrossEntropyLoss(reduction='none')

        # cuda gpu
        if learner_config['gpuid'][0] >= 0: # if using cuda 
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False

        # highest class index from past task
        self.last_valid_out_dim = 0 

        # highest class index from current task
        self.valid_out_dim = 0

        # set up schedules
        self.schedule_type = self.config['schedule_type']
        self.schedule = self.config['schedule']

        # initialize optimizer
        self.init_optimizer()

    def create_model(self):
        pass
    
    def learn_batch(self, train_loader, train_dataset, val_loader  = None, ):
        # train the model for local epochs 

        if self.reset_optimizer:
            self.init_optimizer()
            self.log('Optimizer is reset')
        
        # data weighting for class imbalance
        self.data_weighting(train_dataset)

        # if using validation
        if val_loader is not None:
            self.validation(val_loader)

        for epoch in range(self.config['schedule'][-1]):
            self.epoch = epoch
            if epoch >0 : self.schedular.step()

            for i, (x, y) in enumerate(train_dataset):

                self.model.train() # ensure the model is in training mode

                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                # update model
                loss , output = self.update_model(x,y)

                # TODO add training acc metric
            print(f"epoch {epoch}: {loss.item()}")
        
        self.model.eval() # set the model to the eval mode 

        # self.last_valid_out_dim = self.valid_out_dim
        # self.first_task = False

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

        # for eval
        if self.previous_teacher is not None:
            self.previous_previous_teacher = self.previous_teacher

        # new teacher
        teacher = Teacher(solver=self.model)
        self.previous_teacher = copy.deepcopy(teacher)
        return loss, self.model.state_dict()


    def criterion(self, logits, targets, data_weights):
        """The loss criterion with any additional regularizations added
        The inputs and targets could come from single task 
        The network always makes the predictions with all its heads
        The criterion will match the head and task to calculate the loss.
        Parameters
        ----------
        logits : dict(torch.Tensor)
            Dictionary of predictions, e.g. outs from `forward`
        targets : torch.Tensor
            target labels
        Returns
        -------
        torch._Loss :
            the loss function with any modifications added
        """
        # print('*************')
        # print(data_weights)
        # print('*************')
        loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).mean()
        return loss_supervised    

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        if dw_force is not None:
            dw_cls = dw_force
        elif self.dw:
            dw_cls = self.dw_k[targets.long()]
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        logits = self.forward(inputs)
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), logits

    def visualization(self, dataloader, topdir, name, task, embedding):
        raise NotImplementedError

    def data_visualization(self, dataloader, topdir, name, task):
        raise NotImplementedError

    def cka_eval(self, dataloader):
        raise NotImplementedError

    def validation(self, dataloader, model=None, task_in = None, task_metric='acc', relabel_clusters = True, verbal = True, cka_flag = -1, task_global=False):
        raise NotImplementedError
    

    ##########################################
    #             MODEL UTILS                #
    ##########################################

    # data weighting
    def data_weighting(self, dataset, num_seen=None):

        self.dw_k = torch.tensor(np.ones(self.valid_out_dim + 1, dtype=np.float32))
        # cuda
        if self.cuda:
            self.dw_k = self.dw_k.cuda()

    def save_model(self, filename):
        model_state = self.model.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        self.log('=> Saving class model to:', filename)
        torch.save(model_state, filename + 'class.pth')
        self.log('=> Save Done')

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename + 'class.pth'))
        self.log('=> Load Done')
        if self.gpu:
            self.model = self.model.cuda()
        self.model.eval()

    def load_model_other(self, filename, model):
        model.load_state_dict(torch.load(filename + 'class.pth'))
        if self.gpu:
            model = model.cuda()
        return model.eval()
            

                
    # sets model optimizers
    def init_optimizer(self):

        # parse optimizer args
        print('*****************************************')
        print('*****************************************')
        print('*****************************************')
        print('Num param opt: ' + str(count_parameters(self.model.parameters())))
        self.num_p_train = count_parameters(self.model.parameters())
        print('*****************************************')
        print('*****************************************')
        print('*****************************************')
        optimizer_arg = {'params':self.model.parameters(),
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        
        # create schedules
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)

    

    def forward(self, x):
        return self.model.forward(x)[:, :self.valid_out_dim]

    def predict(self, inputs):
        self.model.eval()
        out = self.forward(inputs)
        return out

    def reset_model(self):
        self.model.apply(weight_reset)

    def print_model(self):
        self.log(self.model)
        self.log('#parameter of model:', self.count_parameter())       
    

    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log("Running on:", device)
        return device
    
    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())   

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def count_parameters(params,model):
    # print("in count _params")
    total_params = sum(p.numel() for p in model.parameters())
    return float(sum(p.numel() for p in params if p.requires_grad) / total_params) * 100.0

def count_parameters_2(model):
    # print("in count _params_2")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())
    print('number of params:', n_parameters)
    print('non trainable parameters', total_parameters)
    print("percent of train paramters ", n_parameters/ total_parameters * 100)

##########################################
#            TEACHER CLASS               #
##########################################

class Teacher(nn.Module):

    def __init__(self, solver):

        super().__init__()
        self.solver = solver

    def generate_scores(self, x, allowed_predictions=None, threshold=None):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x)
        y_hat = y_hat[:, allowed_predictions]

        # set model back to its initial mode
        self.train(mode=mode)

        # threshold if desired
        if threshold is not None:
            # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
            y_hat = F.softmax(y_hat, dim=1)
            ymax, y = torch.max(y_hat, dim=1)
            thresh_mask = ymax > (threshold)
            thresh_idx = thresh_mask.nonzero().view(-1)
            y_hat = y_hat[thresh_idx]
            y = y[thresh_idx]
            return y_hat, y, x[thresh_idx]

        else:
            # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
            ymax, y = torch.max(y_hat, dim=1)

            return y_hat, y

    def generate_scores_pen(self, x):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x=x, pen=True)

        # set model back to its initial mode
        self.train(mode=mode)

        return y_hat

    def generate_scores_layer(self, x, layer):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x=x, pen=True, l = layer)

        # set model back to its initial mode
        self.train(mode=mode)

        return y_hat

def loss_fn_kd(): 
    pass 

def accumulate_acc(output, target, meter, topk):
    meter.update(accuracy(output, target, topk), len(target))
    return meter
