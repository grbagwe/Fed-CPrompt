import os
import sys
import argparse
import torch
import numpy as np
import random
from random import shuffle
from collections import OrderedDict
import dataloaders
from dataloaders.utils import *
from torch.utils.data import DataLoader
import learners
import matplotlib
import copy
import pickle
from copy import deepcopy
from models.vit_coda_p import vit_pt_imnet
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.evaluate_model import evaluate_rounds, evaluate_trained_model
from learners.prompt import DualPrompt
import copy
def average_weights(w):
    """
    https://github.com/AshwinRJ/Federated-Learning-PyTorch
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            # if 'prompt' in key:
            # print(f' {client_data_sample[i],total_data_samples} weight average ratio {client_data_sample[i]/total_data_samples}')
            w_avg[key] += w[i][key]
                          # * client_data_sample[i] / total_data_samples
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def save_model(model, filename):
    model_state = model.state_dict()
    for key in model_state.keys():  # Always save it to cpu
        model_state[key] = model_state[key].cpu()
    print('=> Saving class model to:', filename)
    torch.save(model_state, filename + 'class.pth')
    print('=> Save Done')




class Trainer:

    def __init__(self, args, seed, metric_keys, save_keys):


        self.wandb =args.wandb

        # process inputs
        self.seed = seed # get random seed
        self.num_rounds = args.num_rounds  # get number of rounds in fed
        self.num_clients = 1 # args.num_clients # get number of clients
        self.metric_keys = metric_keys
        self.save_keys = save_keys
        self.vis_flag = args.vis_flag == 1
        self.log_dir = args.log_dir
        self.batch_size = args.batch_size # batch size in clients
        self.workers = args.workers # number of workers in train loader

        # model load directory
        self.model_top_dir = args.log_dir  # model dir 

        # select dataset
        self.grayscale_vis = False
        self.top_k = 1  # acc metric 
        if args.dataset == 'CIFAR10_Fed':
            Dataset = dataloaders.iCIFAR10_Fed
            num_classes = 10
            self.dataset_size = [32, 32, 3]
        elif args.dataset == 'CIFAR100_Fed':
            # print('in cifar 100')
            Dataset = dataloaders.iCIFAR100_Fed  # cifar dataset in continual part
            num_classes = 100
            self.dataset_size = [32, 32, 3]
        # elif args.dataset == 'ImageNet32':
        #     Dataset = dataloaders.iIMAGENETs
        #     num_classes = 100
        #     self.dataset_size = [32, 32, 3]
        # elif args.dataset == 'ImageNet84':
        #     Dataset = dataloaders.iIMAGENETs
        #     num_classes = 100
        #     self.dataset_size = [84, 84, 3]
        # elif args.dataset == 'ImageNet':
        #     Dataset = dataloaders.iIMAGENET
        #     num_classes = 1000
        #     self.dataset_size = [224, 224, 3]
        #     self.top_k = 5
        # elif args.dataset == 'ImageNet_R':
        #     Dataset = dataloaders.iIMAGENET_R
        #     num_classes = 200
        #     self.dataset_size = [224, 224, 3]
        #     self.top_k = 1
        # elif args.dataset == 'ImageNet_D':
        #     Dataset = dataloaders.iIMAGENET_D
        #     num_classes = 200
        #     self.dataset_size = [224, 224, 3]
        #     self.top_k = 1
        # elif args.dataset == 'DomainNet':
        #     Dataset = dataloaders.iDOMAIN_NET
        #     num_classes = 345
        #     self.dataset_size = [224, 224, 3]
        #     self.top_k = 1
        # elif args.dataset == 'TinyImageNet':
        #     Dataset = dataloaders.iTinyIMNET
        #     num_classes = 200
        #     self.dataset_size = [64, 64, 3]
        else:
            raise ValueError(f'{args.dataset} Dataset not implemented!')

        # upper bound flag
        if args.upper_bound_flag:
            args.other_split_size = num_classes  # not using continual part ? 
            args.first_split_size = num_classes  
 
        # load tasks
        class_order = np.arange(num_classes).tolist()
        class_order_logits = np.arange(num_classes).tolist()
        if self.seed > 0 and args.rand_split:
            print('=============================================')
            print('Shuffling....')
            print('pre-shuffle:' + str(class_order))
            if args.dataset == 'ImageNet':
                np.random.seed(1993)
                np.random.shuffle(class_order)
            else:
                random.seed(self.seed)
                random.shuffle(class_order)  # reorder class list
            print('post-shuffle:' + str(class_order))  # new class order 
            print('=============================================')
        self.tasks = []  # task classes in continual class incremtal learnig
        self.tasks_logits = []
        p = 0
        while p < num_classes and (args.max_task == -1 or len(self.tasks) < args.max_task):
            inc = args.other_split_size if p > 0 else args.first_split_size
            self.tasks.append(class_order[p:p + inc])  # create tasks 
            self.tasks_logits.append(class_order_logits[p:p + inc])
            p += inc
        self.num_tasks = len(self.tasks)  # total number of tasks
        self.task_names = [str(i + 1) for i in range(self.num_tasks)]  # task id 

        # number of tasks to perform
        if args.max_task > 0:
            self.max_task = min(args.max_task, len(self.task_names))
        else:
            self.max_task = len(self.task_names)

        # datasets and dataloaders
        k = 1  # number of transforms per image
        if args.model_name.startswith('vit'): # if vit model 
            resize_imnet = True
        else:
            resize_imnet = False
        train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug,
                                                          resize_imnet=resize_imnet)
        test_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=args.train_aug,
                                                         resize_imnet=resize_imnet)
        self.train_dataset = Dataset(args.dataroot, train=True, lab=True, tasks= self.tasks,  transform=train_transform,
                                     seed=self.seed, rand_split=args.rand_split, use_validation=args.validation, num_clients= self.num_clients, iid  = args.iid)
        self.test_dataset = Dataset(args.dataroot, train=False, tasks=self.tasks,
                                    transform=test_transform,
                                    seed=self.seed, rand_split=args.rand_split, use_validation=args.validation, iid  = args.iid)
        # self.complete_test_dataset = copy.deepcopy(self.test_dataset)
        self.add_dim = 0  # add dim to the classifier head 

        # Prepare the self.learner (model)
        self.learner_config = {'num_classes': num_classes,
                               'lr': args.lr,
                               'debug_mode': args.debug_mode == 1, # args debug mode on ? 
                               'momentum': args.momentum,
                               'weight_decay': args.weight_decay,
                               'schedule': args.schedule,
                               'schedule_type': args.schedule_type,
                               'model_type': args.model_type,
                               'model_name': args.model_name,
                               'optimizer': args.optimizer,
                               'gpuid': args.gpuid,
                               'memory': args.memory,
                               'temp': args.temp,
                               'out_dim': num_classes,
                               'overwrite': args.overwrite == 1,
                               'mu': args.mu,
                               'beta': args.beta,
                               'eps': args.eps,
                               'DW': args.DW,
                               'batch_size': args.batch_size,
                               'upper_bound_flag': args.upper_bound_flag,
                               'tasks': self.tasks_logits,
                               'top_k': self.top_k,
                               'prompt_param': [self.num_tasks, args.prompt_param]
                               }
        print(" --------------- task_logits :", self.num_tasks)
        self.learner_type, self.learner_name = args.learner_type, args.learner_name

        self.global_model = vit_pt_imnet(out_dim=num_classes, prompt_flag = 'codap', prompt_param=self.learner_config['prompt_param'])

        self.client_learner  = {}
        self.global_learner = DualPrompt(learner_config=self.learner_config,model= copy.deepcopy(self.global_model))
        # for i in range(self.num_clients):
        #     self.client_learner[i] = copy.deepcopy(DualPrompt(learner_config=self.learner_config,model= copy.deepcopy(self.global_model)))
        # # self.learner.print_model()
        # print("\n\n\n args:", args, "\n\n\nlearner config :", self.learner_config)
        # self.global_model = torch.nn.DataParallel(self.global_model, device_ids=self.learner_config['gpuid'])

        self.wandb.watch(self.global_model, log="all")



    def train(self, avg_metrics):

        self.global_model = torch.nn.DataParallel(self.global_model, device_ids=self.learner_config['gpuid']) # set model to dataparallel on multi gpu's
        self.global_model.cuda()



        # temporary results saving
        
        temp_table = {}
        
        # self.all_clients = np.arange(0, self.num_clients)

        for mkey in self.metric_keys: temp_table[mkey] = []
        temp_dir = self.log_dir + '/csv/'
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)

        # for each task
        acc_matrix = np.zeros((self.num_tasks, self.num_tasks))

        for task_no in range(self.max_task): # cycle through the tasks # each task

            # TODO 
            if task_no > 0: # freeze params on the global model # not required
                try:
                    if self.global_model.module.prompt is not None:
                        self.global_model.module.prompt.process_frequency()
                except:
                    if self.global_model is not None:
                        self.global_model.prompt.process_frequency()

            task = self.tasks_logits[task_no] # current task idxs # 
            self.add_dim = len(task)

            # save current task index
            self.current_t_index = task_no



            # # save name for learner specific eval
            # if self.vis_flag:
            #     vis_dir = self.log_dir + '/visualizations/task-' + self.task_names[i] + '/'
            #     if not os.path.exists(vis_dir): os.makedirs(vis_dir)
            # else:
            #     vis_dir = None

            # set seeds
            random.seed(self.seed * 100 + task_no )
            np.random.seed(self.seed * 100 + task_no)
            torch.manual_seed(self.seed * 100 + task_no)
            torch.cuda.manual_seed(self.seed * 100 + task_no)

            # print name
            train_name = self.task_names[task_no]  # task name 0, 1 ,2 ... 
            print('======================', train_name, '=======================')

            # load dataset for task
 



            # add valid class to classifier
            # print("line 309 ", self.add_dim)
            # local_weights = {}
            # if i == 0:
            #     for client in range(self.num_clients):
            #         self.local_learner[client] = deepcopy(self.global_learner)
                # self.local_learner[client]['learner'] = deepcopy(self.global_learner)
                # self.local_learner[client]['model'] = deepcopy(self.global_learner.model)


            # for round_ in range(self.num_rounds):
                # L_subset_clients = np.random.choice(range(self.num_clients),
                #                                     self.num_clients, replace=False
                #                                     )
                # L_subset_clients = np.random.choice(range(self.num_clients),
                #                                     3, replace=False
                #                                     )

                # # for client in L_subset_clients:
                # for client in range(int(1)):

            try:
                self.global_learner.model.module.task_id = task_no
            except:
                self.global_learner.model.task_id = task_no
            
            self.global_learner.last_valid_out_dim = self.global_learner.valid_out_dim

            self.global_learner.add_valid_output_dim(self.add_dim)



                # if round_ == 0:
                #     self.global_learner.add_valid_output_dim(self.add_dim)
                #     if task_no> 0 :
                #         try:
                #             if self.global_learner.model.module.prompt is not None:
                #                 self.global_learner.model.module.prompt.process_frequency()
                                
                #         except:
                #             if self.client_learner[client].model is not None:
                #                 self.client_learner[client].model.prompt.process_frequency()
                # else:
                    # self.local_learner[client].reset_optimizer = False
                    # self.client_learner[client].model.load_state_dict(self.global_model.state_dict())



            self.train_dataset.load_dataset(t=task_no, train=True, client= 0) # load client dataset
            print(np.unique(self.train_dataset.targets))
                


                # set task id for model (needed for prompting)

                # load dataloader
            if len(self.train_dataset.targets) < self.batch_size: # if dataste is smaller than the batch size in niid clients
                temp_batch_size = int(len(self.train_dataset.targets)/2)
                train_loader = DataLoader(self.train_dataset, batch_size=temp_batch_size, shuffle=True, drop_last=True,
                                        num_workers=int(self.workers)) # create train loader
            else:
                train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                        num_workers=int(self.workers))
                
            model_save_dir_client = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+ \
                '/task-'+str(self.task_names[task_no])+'/' + 'after training-' + '/'
            if not os.path.exists(model_save_dir_client): os.makedirs(model_save_dir_client)

            

            # learn

            print( f" ----------------------- task : {task_no} -----------------")
            model_save_dir_p = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+ \
                '/task-'+str(self.task_names[task_no -1])  + '/' + 'after testing' + '/' + 'class.pth'
            print(model_save_dir_p)
                # a = copy.deepcopy(self.client_learner[client].model.state_dict())

            avg_train_time, lweights = self.global_learner.learn_batch(train_loader, self.train_dataset) # learn the batch
            
            # save model
            
            # if  os.path.exists(model_save_dir_p):
            #     print('overwriting model dimenson ', len(task) * task_no)
            #     p_model_weight = torch.load(model_save_dir_p)
            #     lweights['module.last.weight'][:len(task) * task_no] = p_model_weight['module.last.weight'][:len(task) * task_no].cuda()
            #     lweights['module.last.bias'][:len(task) * task_no] = p_model_weight['module.last.bias'][:len(task) * task_no].cuda()
            #     self.global_learner.model.state_dict()['module.last.weight'][:len(task) * task_no] = p_model_weight['module.last.weight'][:len(task) * task_no].cuda()
            #     self.global_learner.model.state_dict()['module.last.bias'][:len(task) * task_no] = p_model_weight['module.last.bias'][:len(task) * task_no].cuda()
            # else: print('not in overwriting')

            self.global_model.load_state_dict(lweights)
            self.global_learner.model = self.global_model


            self.global_learner.save_model(model_save_dir_client)                   
            
                # ll = 0
                # for n,p in self.client_learner[0].model.named_parameters():
                #     ll += torch.sum(a[str(n)] - lweights[str(n)])
                # print("ll ", ll)
                # local_weights[client] = copy.deepcopy(lweights)
                # local_weights[client]  = self.local_learner[client].model.state_dict()
            # local_weights = []

                # average model after round 
                # for client in range(self.num_clients):
                #     local_weights.append(self.local_learner[client].model.state_dict())
                # print("local_weights : ", local_weights.shape)
                # avg_weights = average_weights(local_weights)
                # with open('local_weights.pkl', 'wb') as file:

                #     pickle.dump(local_weights,file)
                
            


            model_save_dir_global = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+ \
                '/task-'+str(self.task_names[task_no])+'/' + 'global' + '/'
            if not os.path.exists(model_save_dir_global): os.makedirs(model_save_dir_global)
            save_model(self.global_model, model_save_dir_global)



            self.test_dataset.load_dataset(task_no, train=True)

            test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
            batch_loss, batch_acc1, batch_acc5 = evaluate_rounds(self.global_model, test_dataloader,  task_no = task_no,
                                                                test_dataset= None, valid_out_dim = (task_no+1) * 10, 
                                                                round = 0, gpu= True, wandb = self.wandb)

            # writer.add_scalar(tag=f"{task_no}loss", scalar_value=batch_loss, global_step=round_)
            # writer.add_scalar(tag=f"{task_no}acc1", scalar_value=batch_acc1, global_step=round_)
            test_stats, forgetting = evaluate_trained_model(self.global_model, self.test_dataset, (task_no+1) * 10, task_no=task_no, acc_matrix=acc_matrix, gpu= True, num_tasks = self.num_tasks,wandb= self.wandb)
        
            # after the rounds in current task
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+ \
                '/task-'+str(self.task_names[task_no]) + '/' + 'after testing' + '/'
            if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
            save_model(self.global_model, model_save_dir)
            # save model
            # self.global_model.save_model(model_save_dir)

        

        return avg_metrics

    def summarize_acc(self, acc_dict, acc_table, acc_table_pt):

        # unpack dictionary
        avg_acc_all = acc_dict['global']
        avg_acc_pt = acc_dict['pt']
        avg_acc_pt_local = acc_dict['pt-local']

        # Calculate average performance across self.tasks
        # Customize this part for a different performance metric
        avg_acc_history = [0] * self.max_task
        for i in range(self.max_task):
            train_name = self.task_names[i]
            cls_acc_sum = 0
            for j in range(i + 1):
                val_name = self.task_names[j]
                cls_acc_sum += acc_table[val_name][train_name]
                avg_acc_pt[j, i, self.seed] = acc_table[val_name][train_name]
                avg_acc_pt_local[j, i, self.seed] = acc_table_pt[val_name][train_name]
            avg_acc_history[i] = cls_acc_sum / (i + 1)

        # Gather the final avg accuracy
        avg_acc_all[:, self.seed] = avg_acc_history

        # repack dictionary and return
        return {'global': avg_acc_all, 'pt': avg_acc_pt, 'pt-local': avg_acc_pt_local}

    def evaluate(self, avg_metrics):

        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

        # store results
        metric_table = {}
        metric_table_local = {}
        for mkey in self.metric_keys:
            metric_table[mkey] = {}
            metric_table_local[mkey] = {}

        for i in range(self.max_task):

            # load model
            model_save_dir = self.model_top_dir + '/models/repeat-' + str(self.seed + 1) + '/task-' + self.task_names[
                i] + '/'
            self.learner.task_count = i
            self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
            self.learner.pre_steps()
            self.learner.load_model(model_save_dir)

            # frequency table process
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_frequency()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_frequency()

            # evaluate acc
            metric_table['acc'][self.task_names[i]] = OrderedDict()
            metric_table_local['acc'][self.task_names[i]] = OrderedDict()
            self.reset_cluster_labels = True
            for j in range(i + 1):
                val_name = self.task_names[j]
                metric_table['acc'][val_name][self.task_names[i]] = self.task_eval(j)
            for j in range(i + 1):
                val_name = self.task_names[j]
                metric_table_local['acc'][val_name][self.task_names[i]] = self.task_eval(j, local=True)

            # evaluate aux_task
            metric_table['aux_task'][self.task_names[i]] = OrderedDict()
            metric_table_local['aux_task'][self.task_names[i]] = OrderedDict()
            for j in range(i + 1):
                val_name = self.task_names[j]
                metric_table['aux_task'][val_name][self.task_names[i]] = self.task_eval(j, task='aux_task')
                metric_table_local['aux_task'][val_name][self.task_names[i]] = self.task_eval(j, local=True,
                                                                                              task='aux_task')

        # summarize metrics
        avg_metrics['acc'] = self.summarize_acc(avg_metrics['acc'], metric_table['acc'], metric_table_local['acc'])
        avg_metrics['aux_task'] = self.summarize_acc(avg_metrics['aux_task'], metric_table['aux_task'],
                                                     metric_table_local['aux_task'])

        return avg_metrics
