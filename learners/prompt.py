from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
# import models
from models.vision_transformer import VisionTransformer
# from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
import scipy as sp
import scipy.linalg as linalg
# from utils.visualization import tsne_eval, confusion_matrix_vis, pca_eval
from torch.optim import Optimizer
import contextlib
import os
from .default import NormalNN, loss_fn_kd, Teacher, count_parameters, count_parameters_2, accumulate_acc
import copy
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.schedulers import CosineSchedule
from torch.autograd import Variable, Function
from utils.metric import accuracy, AverageMeter, Timer

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1) * 1.0e3  # scale up
        distance_negative = (anchor - negative).pow(2).sum(1) * 1.0e1  # scale up
        # print(distance_positive)
        losses = torch.relu((distance_positive - distance_negative).sum() + self.margin)
        return losses.mean()
class positive_loss(torch.nn.Module):
    def __init__(self, margin=0.0):
        super(positive_loss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive):
        distance_positive = (anchor - positive).pow(2).sum(1) * 1.0e3 # scale up
        losses = torch.relu((distance_positive).sum() + self.margin)
        return losses.mean()
    

class LWF(NormalNN):

    def __init__(self, learner_config, model) -> None:
        super(LWF, self).__init__(learner_config, model)
        self.previous_previous_teacher = None
        self.past_tasks =[]
        self.first_task  = True
        self.first_block = True
        self.ce_loss  = nn.BCELoss()
        self.init_task_param_reg = self.eps > 0
        self.previous_model = None
        self.cos=torch.nn.CosineSimilarity(dim=-1) # cosine similarity


    def learn_batch(self, train_loader, train_dataset, val_loader=None, global_model =None, previous_task_model= None):

        # L2 from the start
        if self.init_task_param_reg: self.accumulate_block_memory(train_loader)
        # print("before training", self.model.state_dict()['module.last.weight'])
        # init teacher
        # if self.previous_teacher is None:
        #     teacher = Teacher(solver=self.model)
        #     self.previous_teacher = copy.deepcopy(teacher)
        try:
            print("---size of the batch ", len(train_dataset.data),"")
        except:
            print("---size of the batch ", len(train_dataset.dataset.data),"")
        # train
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()
        # if need_train:
        #
        #     # data weighting
        self.data_weighting(train_dataset)
            
            # Evaluate the performance of current task
        if val_loader is not None:
            self.validation(val_loader)
        
        # global weight collector for fed prox
        global_weight_collector = list(global_model.parameters())
        
        # training metrics 
        

        for epoch in range(self.config['schedule'][-1]):
            self.epoch=epoch
            losses = [AverageMeter() for l in range(4)]
            acc = AverageMeter()
            batch_time = AverageMeter()

            if epoch > 0: self.scheduler.step()

            for i, (x, y)  in enumerate(train_loader):
                self.step = i

                # verify in train mode
                self.model.train()

                # send data to gpu
                if self.gpu:
                    x = x.cuda()
                    y = y.cuda()

                # model update - training data
                loss, loss_class, loss_distill, output, fedmoonLoss = self.update_model(x, y, global_model = global_model, previous_task_model = previous_task_model)

                accumulate_acc(output, y, acc, topk=(self.top_k,))
                losses[0].update(loss,  y.size(0)) 
                losses[1].update(loss_class,  y.size(0)) 
                losses[2].update(loss_distill,  y.size(0))
                losses[3].update(fedmoonLoss,  y.size(0))

                # measure accuracy and record loss
                y = y.detach()

            # eval update
            print(f"epoch {epoch}: {loss.item()}")
            self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
            self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f} | C2 loss {lossp.avg:.3f}'.format(loss=losses[0],acc=acc,lossp=losses[3]))
            # self.log(' * Class Loss {loss.avg:.3f} | KD Loss {lossb.avg:.3f}'.format(loss=losses[1],lossb=losses[2]))
            
            # Evaluate the performance of current task
            if val_loader is not None:
                self.validation(val_loader)

            # reset
        # print("after training", self.model.state_dict())
        # print("after training", self.model.state_dict()['module.last.weight'])

        print("Saving previous model")
        self.previous_model = copy.deepcopy(self.model)
        self.model.eval()
        torch.save(self.previous_model.state_dict(), "previous_modelmodel.pth")
    

        return loss.item(), self.model.state_dict() , acc.avg


    def update_model(self, inputs, targets, target_KD = None, global_model = None):
        
        total_loss = torch.zeros((1,), requires_grad=True).cuda()

        if self.dw:
            dw_cls = self.dw_k[targets.long()]
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        logits = self.forward(inputs)
        loss_class = self.criterion(logits, targets.long(), dw_cls)

        total_loss += loss_class

        # KD
        if target_KD is not None:
            dw_KD = self.dw_k[-1 * torch.ones(len(target_KD),).long()]
            logits_KD = logits
            loss_distill = loss_fn_kd(logits_KD, target_KD, dw_KD, np.arange(self.last_valid_out_dim).tolist(), self.DTemp)
            total_loss += self.mu * loss_distill
        else:
            loss_distill = torch.zeros((1,), requires_grad=True).cuda()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), loss_class.detach(), loss_distill.detach(), logits




class DualPrompt(LWF):

    def __init__(self, learner_config, model):
        self.prompt_param = learner_config['prompt_param']
        super(DualPrompt, self).__init__(learner_config, model)
        self.triplet_loss = TripletLoss(margin=1)
        


    # update model - add dual prompt loss   
    def update_model(self, inputs, targets, target_KD = None,  global_model = None, previous_task_model = None):
        fedmoonLoss = torch.zeros((1,), requires_grad=True).cuda()
        self.optimizer.zero_grad()
        t_c2loss = torch.zeros((1,), requires_grad=True).cuda()
        
        # optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.005)
        # logits
        logits, prompt_loss, prelogits_current = self.model(inputs, train=True) # original model
        # print("prompt_loss", logits)
        if self.fedmoon[0] == 2:
            with torch.no_grad(): 
                _, _, prelogits_global = global_model(inputs, train=True)
            posi = self.cos(prelogits_current, prelogits_global)
            if self.previous_model is not None:
                with torch.no_grad():
                    _, _, prelogits_previous = self.previous_model(inputs, train=True)
                nega = self.cos(prelogits_current, prelogits_previous)
                # print("posi", posi)
                # print("nega", nega)
                numerator = torch.exp(posi/self.tau)
                denominator = torch.exp(posi/self.tau) + torch.exp(nega/self.tau)
                fedmoonLoss = -torch.log(numerator/denominator)
                # print ("fedmoonLoss", fedmoonLoss)
                # print("numerator", numerator)
                # print("denominator", denominator)
                fedmoonLoss = fedmoonLoss.mean()
        elif self.fedmoon[0] == 1:
            with torch.no_grad(): 
                _, _, prelogits_global = global_model(inputs, train=True)
            posi = self.cos(prelogits_current, prelogits_global) # decrease the distance between current task and global model
            if previous_task_model is not None:
                with torch.no_grad():
                    _, _, prelogits_previous_task = previous_task_model(inputs, train=True)
                nega = self.cos(prelogits_current, prelogits_previous_task) # increase the distance between current task and previous task model
                numerator = torch.exp(posi/self.tau)
                denominator = torch.exp(posi/self.tau) + torch.exp(nega/self.tau)
                fedmoonLoss = -torch.log(numerator/denominator)
                fedmoonLoss = fedmoonLoss.mean()
        elif self.fedmoon[0] == 3:                
            # posi = self.cos(prelogits_current, prelogits_global) # decrease the distance between current task and global model
            # initialize nega1 and nega2 as torch like prelogits_current
            # if previous_task_model is not None and  self.previous_model is not None:
            if previous_task_model is not None:
                with torch.no_grad():
                    _, _, prelogits_previous_task = previous_task_model(inputs, train=True)
                    _, _, prelogits_global = global_model(inputs, train=True)
                _, _, prelogits_previous_task = previous_task_model(inputs, train=True)
                t_c2loss =self.triplet_loss(prelogits_current, prelogits_global, prelogits_previous_task)
            if previous_task_model is None:
                with torch.no_grad():
                    _, _, prelogits_global = global_model(inputs, train=True)
                t_c2loss = positive_loss()(prelogits_current, prelogits_global)


                # with torch.no_grad():
                #     _, _, prelogits_previous_task = previous_task_model(inputs, train=True)
                # with torch.no_grad():
                #     _, _, prelogits_previous = self.previous_model(inputs, train=True)
                # nega1 = self.cos(prelogits_current, prelogits_previous_task) # increase the distance between current task and previous task model
                # nega2 = self.cos(prelogits_current, prelogits_previous) # increase the distance between current task and previous client model

                # numerator = torch.exp(posi/self.tau)
                # denominator = torch.exp(posi/self.tau) + ( ( torch.exp(nega1/self.tau) + torch.exp(nega2/self.tau) ) / 2 )
                # fedmoonLoss = -torch.log(numerator/denominator)
                # fedmoonLoss = fedmoonLoss.mean()

                # constrastive loss on the prompts in the model
                
                # task_no = (self.valid_out_dim // 10) -1
                # # print("task_no", task_no)
                # # iterate over all the layers in the model
                # m_count = 0
                # for key, param in self.model.named_parameters():
                #     t_nega, t_posi,   = 0 , 0
                #     if 'prompt' in key: 
                #         p1 = self.model.state_dict()[key] # get far from the promtps  model
                #         p2 = global_model.state_dict()[key] # positive model

                # #         print(key)
                #         for i in range(task_no):
                #         #     print(p1[i*10:i*10+10].shape,i*10,i*10+10 )
                #             nega = self.cos(p1[i*10:i*10+10], p1[task_no*10:task_no*10+10])
                # #             print(nega.shape)
                # #             print(nega.mean())
                #             t_nega += torch.abs(nega)
                # #             print("t", t_nega)


                # #         print(key)
                # #         for i in range(task_no -1):
                #         #     print(p1[i*10:i*10+10].shape,i*10,i*10+10 )
                #         posi = self.cos(p2[task_no*10:task_no*10+10], p1[task_no*10:task_no*10+10])

                # #             print(nega.mean())
                #         t_posi += posi

                #         if self.previous_model  is not None:
                #             p3 =  self.previous_model.state_dict()[key]
                #             nega2 = self.cos(p3[(task_no)*10:task_no*10+10], p1[(task_no)*10:task_no*10+10]) # far from the prompts of the previous task
                #             nega2 = nega2
                #         else :
                #             nega2 = torch.zeros_like(t_posi) # far from the prompts of the previous task
                #     #         print("t", t_posi)
                #         # print(t_nega, t_posi, nega2)

                #         numerator = torch.exp(t_posi/self.tau)
                #         denominator = torch.exp(t_posi/self.tau) + (torch.exp(t_nega/self.tau))
                #         c2loss = -torch.log(numerator/denominator)
                #         numerator2 = torch.exp(t_posi/self.tau)
                #         denominator2 = torch.exp(t_posi/self.tau) + (torch.exp(nega2/self.tau))
                #         c2loss2 = -torch.log(numerator2/denominator2)
                #         t_c2loss += c2loss.mean() + c2loss2.mean()
                #         print("tposi", t_posi, t_nega, nega2, c2loss.mean(), c2loss2.mean())
                #         # print(t_c2loss.mean())
                #         m_count += 1
                #     # print('posi', posi, 'nega', nega, 'nega2', nega2, 't_posi', t_posi.mean(), 't_nega', t_nega.mean(), 't_c2loss', t_c2loss.mean())
                # # print(key, p3[task_no*10:task_no*10+10], p1[task_no*10:task_no*10+10], p2[task_no*10:task_no*10+10])
                # t_c2loss = t_c2loss / m_count
                # outputs_current = logits[:,:self.valid_out_dim]
                # outputs_global = torch.zeros_like(outputs_current)
                # outputs_previous = torch.zeros_like(outputs_current)
                # outputs_previous_task = torch.zeros_like(outputs_current)
                # with torch.no_grad():

                #     if global_model is not None: # close to this 
                #         outputs_global , _, _ = global_model(inputs, train=True) # positive 
                #         outputs_global = outputs_global[:,:self.valid_out_dim]
                #     if self.previous_model is not None: # far from this 
                #         outputs_previous , _, _ = self.previous_model(inputs, train=True) # nega 1 
                #         outputs_previous = outputs_previous[:,:self.valid_out_dim]
                #     if previous_task_model is not None:
                #         outputs_previous_task , _, _ = previous_task_model(inputs, train=True) # nega 2
                #         outputs_previous_task = outputs_previous_task[:,:self.valid_out_dim]

                #     nega1 = self.cos(outputs_current, outputs_previous)
                #     nega2 = self.cos(outputs_current, outputs_previous_task)
                #     posi = self.cos(outputs_current, outputs_global)
                #     num1 = torch.exp(posi/self.tau)
                #     den1 = torch.exp(posi/self.tau) + (torch.exp(nega1/self.tau))
                #     c2loss1 = -torch.log(num1/den1)
                #     num2 = torch.exp(posi/self.tau)
                #     den2 = torch.exp(posi/self.tau) + (torch.exp(nega2/self.tau))
                #     c2loss2 = -torch.log(num2/den2)
                #     t_c2loss = c2loss1.mean() + c2loss2.mean()

                # print("nega1", nega1.shape, "nega2", nega2.shape, "posi", posi.shape, outputs_global.shape, outputs_current.shape, outputs_previous.shape, outputs_previous_task.shape)

                


        
        # print(self.muMoon * t_c2loss)
        # print first 4 values of model parameters
        # print('model parameters', self.model.state_dict()['module.prompt.e_p_0'][0:4])
        # for key, param in self.model.named_parameters():
        #     if 'prompt' in key:
        #         print(key, param[0:4])    

            
        logits = logits[:,:self.valid_out_dim]
        # print(logits)

        # # bce
        # target_mod = get_one_hot(targets-self.last_valid_out_dim, self.valid_out_dim-self.last_valid_out_dim)
        # total_loss = self.ce_loss(torch.sigmoid(logits[:,self.last_valid_out_dim:self.valid_out_dim]), target_mod)

        # standard ce
        # print("self.last_valid_out_dim", self.last_valid_out_dim)
        # print("logits dim ", logits.shape)
        logits[:,:self.last_valid_out_dim] = -float('inf') # to freeze the part of the last layer




        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)
        # print("t_c2loss", t_c2loss)

        # print(torch.argmax(logits, dim = 1), targets.long())
        # ce loss
        total_loss = total_loss  +  t_c2loss # mu is the self.muMoon * +  self.muMoon * fedmoonLoss
        # for fed prox
        # fed_prox_reg = 0.0
        # if self.muProx > 0 and global_weight_collector is not None:
        #     for param_index, param in enumerate(self.model.parameters()):
        #         fed_prox_reg += ((self.muProx / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
        #     total_loss += fed_prox_reg # add fed prox reg
        # else: 
            # raise ValueError("muProx is not set or global_weight_collector is None")

        # add fed moon
        # for fed moon we need output from the second last layer of the net
        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        fedmoonLoss =  t_c2loss


        return total_loss.detach(), prompt_loss.sum().detach(), torch.zeros((1,), requires_grad=True).cuda().detach(), logits , fedmoonLoss.detach()

    # sets model optimizers
    def init_optimizer(self):

        # parse optimizer args
        params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.last.parameters())
        # print('Num param opt: ' + str(count_parameters(params_to_opt, self.model)))
        # print('param opt: ' + str(count_parameters_2(self.model)))
        self.num_p_train = count_parameters(params_to_opt, self.model)
        optimizer_arg = {'params':params_to_opt,
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

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        # self.model.prompt = self.model.prompt.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

class L2P(DualPrompt):

    def __init__(self, learner_config, model ):
        super(L2P, self).__init__(learner_config, model)




