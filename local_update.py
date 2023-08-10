import torch, torchvision, timm
import numpy as np
import torch.nn as nn
import timm
from torch.utils.tensorboard import SummaryWriter
import argparse
import os

# import math
#
# import numpy as np
# import torch
# import torchvision
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# import random
# import matplotlib.pyplot as plt
# import pickle
# import os
# from torch.utils.data import DataLoader
# from arguments_variables import *
# from create_dataset import *
# import evaluate_model
# import copy
# import updates
# import models
# import utils
# from timm.models import create_model
# from timm.scheduler import create_scheduler
# from timm.optim import create_optimizer
# from easydict import EasyDict
# import torch.backends.cudnn as cudnn
# import sys
#
# from pathlib import Path
# from timm.optim import create_optimizer
from tqdm import tqdm
import wandb
# from torch.utils.tensorboard import SummaryWriter
# from torch.nn.parallel import DistributedDataParallel as DDP

def get_args_parser():
    parser = argparse.ArgumentParser('DualPrompt CIFAR-100 training and evaluation configs', add_help=False)

    parser.add_argument('--wandb_group', default=' ', type=str, help = "group for wandb runs" )
    parser.add_argument('--batch-size', default=24, type=int, help='Batch size per device')
    parser.add_argument('--epochs', default=5, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--pretrained', default=True, help='Load pretrained model or not')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.0, metavar='PCT', help='Drop path rate (default: 0.)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: (0.9, 0.999), use opt default)')
    parser.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay (default: 0.0)')
    parser.add_argument('--reinit_optimizer', type=bool, default=True, help='reinit optimizer (default: True)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "constant"')
    parser.add_argument('--lr', type=float, default=0.03, metavar='LR', help='learning rate (default: 0.03)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--unscale_lr', type=bool, default=False, help='scaling lr by batch size (default: False)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')

    # Data parameters
    parser.add_argument('--data-path', default='/local_datasets/', type=str, help='dataset path')
    parser.add_argument('--dataset', default='CIFAR100', type=str, help='dataset name')
    parser.add_argument('--iid_type', default='homo', type=str,
                        help='type of data heteroginity [homo, label_1, label_2, noniid_label_dir')
    # parser.add_argument('--iid', default=False, type=bool)
    parser.add_argument('--shuffle', default=True, help='shuffle the data order')
    parser.add_argument('--output_dir', default='./output', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--experiment_name', default='5', type=str, help='experiment number')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Continual learning parameters
    parser.add_argument('--num_tasks', default=10, type=int, help='number of sequential tasks')
    parser.add_argument('--classes_per_task', default=10, type=int, help='number of classes per task')
    parser.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')
    parser.add_argument('--task_inc', default=False, type=bool, help='if doing task incremental')
    parser.add_argument('--temp', default=False, type=bool, help='temporay setting to debug')

    # G-Prompt parameters
    parser.add_argument('--use_g_prompt', default=True, type=bool, help='if using G-Prompt')
    parser.add_argument('--g_prompt_length', default=5, type=int, help='length of G-Prompt')
    parser.add_argument('--g_prompt_layer_idx', default=[0, 1], type=int, nargs="+",
                        help='the layer index of the G-Prompt')
    parser.add_argument('--use_prefix_tune_for_g_prompt', default=True, type=bool,
                        help='if using the prefix tune for G-Prompt')

    # E-Prompt parameters
    parser.add_argument('--use_e_prompt', default=True, type=bool, help='if using the E-Prompt')
    parser.add_argument('--e_prompt_layer_idx', default=[2, 3, 4], type=int, nargs="+",
                        help='the layer index of the E-Prompt')
    parser.add_argument('--use_prefix_tune_for_e_prompt', default=True, type=bool,
                        help='if using the prefix tune for E-Prompt')

    # Use prompt pool in L2P to implement E-Prompt
    parser.add_argument('--prompt_pool', default=True, type=bool, )
    parser.add_argument('--size', default=10, type=int, )
    parser.add_argument('--length', default=5, type=int, )
    parser.add_argument('--top_k', default=1, type=int, )
    parser.add_argument('--initializer', default='uniform', type=str, )
    parser.add_argument('--prompt_key', default=True, type=bool, )
    parser.add_argument('--prompt_key_init', default='uniform', type=str)
    parser.add_argument('--use_prompt_mask', default=True, type=bool)
    parser.add_argument('--mask_first_epoch', default=False, type=bool)
    parser.add_argument('--shared_prompt_pool', default=True, type=bool)
    parser.add_argument('--shared_prompt_key', default=False, type=bool)
    parser.add_argument('--batchwise_prompt', default=True, type=bool)
    parser.add_argument('--embedding_key', default='cls', type=str)
    parser.add_argument('--predefined_key', default='', type=str)
    parser.add_argument('--pull_constraint', default=True)
    parser.add_argument('--pull_constraint_coeff', default=1.0, type=float)
    parser.add_argument('--same_key_value', default=False, type=bool)

    # ViT parameters
    parser.add_argument('--global_pool', default='token', choices=['token', 'avg'], type=str,
                        help='type of global pooling for final sequence')
    parser.add_argument('--head_type', default='token', choices=['token', 'gap', 'prompt', 'token+prompt'], type=str,
                        help='input type of classification head')
    parser.add_argument('--freeze', default=['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'], nargs='*',
                        type=list, help='freeze part in backbone model')

    # fed learning parameters
    parser.add_argument('--no_total_classes', type=int, default=100, help='The frequency of printing')
    parser.add_argument('--num_rounds', type=int, default=10, help='The number of global rounds')
    parser.add_argument('--local_epochs', type=int, default=5, help='The number of global rounds')
    parser.add_argument('--beta', type=float, default=0.5, help='The number of global rounds')
    parser.add_argument('--algo', default='fed_avg', type=str, help='type of algorithms ')
    parser.add_argument('--loss_type', default='l2p_loss', type=str, help='type of loss func l2p_loss FCL_loss ')

    parser.add_argument('--subset_ratio', type=float, default=0.3, help='subset client fractions ')
    parser.add_argument('--no_total_clients', type=int, default=10, help='total number clients ')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')

    # Misc parameters
    parser.add_argument('--print_freq', type=int, default=10, help='The frequency of printing')
    parser.add_argument('--prompt_idx', default=True, type=bool)
    parser.add_argument('--test_speed', action='store_true', help='whether to measure throughput of model')
    parser.add_argument('--only_test_speed', action='store_true', help='only measure throughput of model')
    parser.add_argument('--pacing_clients', type=float, default=0.0, help='percent of subset clients that pace')

    # wandb parameters
    parser.add_argument('--group', type=str,default = "FCL",  help='group name to sort the experiment')




    # # AdaptFormer related parameters
    # parser.add_argument('--ffn_adapt', default=False, type=bool , help='whether activate AdaptFormer')
    # parser.add_argument('--ffn_num', default=64, type=int, help='bottleneck middle dimension')
    # parser.add_argument('--vpt', default=False, action='store_true', help='whether activate VPT')
    # parser.add_argument('--vpt_num', default=1, type=int, help='number of VPT prompts')
    # parser.add_argument('--fulltune', default=False, action='store_true', help='full finetune model')
    # parser.add_argument('--useadapter', default=False, type = bool, help='wherather to use adapter settings')

    return parser





def main(args):
    # init distributed
    # utils.init_distributed_mode(args)
    args.distributed = False
    device = torch.device(args.device)

    print(f'this is in main {args.iid_type} , data heteroginity \n\n')
    # print(args)
    # args.use_g_prompt = False
    # args.use_prefix_tune_for_g_prompt = False

    # number of subset clients
    args.no_subset_clients = int(args.subset_ratio * args.no_total_clients)
    K_total_clients = list(range(0, args.no_total_clients))

    args.nb_classes = args.no_total_classes
    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.no_total_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    FCL_dir = f'../datasets/FCL_data/{args.dataset}/{args.experiment_name}'
    FCL_save_dir = f'../experiment_runs/{args.dataset}/{args.experiment_name}'
    args.dir = FCL_dir
    args.save_dir = FCL_save_dir
    os.makedirs(FCL_save_dir, exist_ok=True)
    if not os.path.exists(FCL_dir):
        train_dataset, test_dataset, class_mask = make_client_datasets(args.dataset, args.experiment_name, args)
    else:
        train_dataset, test_dataset = get_dataset(args.dataset)
        file_name = f'{FCL_dir}/' + '/class_mask' + '.pkl'
        infile = open(file_name, 'rb')
        class_mask = pickle.load(infile)
    # test_dataset = f'{FCL_dir}/test/'

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.no_total_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
        use_g_prompt=args.use_g_prompt,
        g_prompt_length=args.g_prompt_length,
        g_prompt_layer_idx=args.g_prompt_layer_idx,
        use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
        use_e_prompt=args.use_e_prompt,
        e_prompt_layer_idx=args.e_prompt_layer_idx,
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
        same_key_value=args.same_key_value,
    )

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original_model.to(device)
    model.to(device)
    if args.freeze:
        # all parameters are frozen for original vit model
        for p in original_model.parameters():
            p.requires_grad = False

        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

    print(args)
    print(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original_model.to(device)
    model.to(device)

    # model_without_ddp = model
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module

    model.train()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    print('non trainable parameters', sum(p.numel() for p in model.parameters()))
    # for n,p in model.named_parameters():
    #     print(n)
    # randomly init the prompt parameters
    # loop over each tasks

    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    if args.unscale_lr:
        global_batch_size = args.batch_size * args.world_size
    else:
        global_batch_size = args.batch_size
    args.lr = args.lr * global_batch_size / 256.0

    '''
    wandb  logging
    '''
    writer = SummaryWriter()
    args.writer = writer
    run = wandb.init(project="Federated_Continual_Learning_with_Prompts",
                     sync_tensorboard=True, group= args.wandb_group)
    writer = SummaryWriter(log_dir = f"{run.dir}")

    wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config  # Initialize config
    config.args = args  # input batch size for training (default: 64)
    config.test_batch_size = args.batch_size  # input batch size for testing (default: 1000)
    config.epochs = args.num_rounds  # number of epochs to train (default: 10)
    config.lr = args.lr  # learning rate (default: 0.01)
    config.subset_ratio = args.subset_ratio
    config.num_total_clients = args.no_total_clients
    config.data_type = args.iid_type
    config.num_tasks = args.num_tasks
    config.prompt_length = args.length
    config.pool_size = args.prompt_pool
    config.top_k = args.top_k
    config.algo = args.algo
    # config.momentum = 0.1  # SGD momentum (default: 0.5)
    # config.no_cuda = False  # disables CUDA training
    # config.seed = 42  # random seed (default: 42)
    # config.log_interval = 10  # how many batches to wait before logging training status

    # test_dataset = f'{FCL_dir}/test/'
    pickle_model = copy.deepcopy(model)
    wandb.watch(model, log="all")
    args.wandb = wandb
    for task_no in range(0, args.num_tasks):

        file_name = f'{FCL_dir}/train/' + '/task_' + str(task_no) + '.pkl'
        infile = open(file_name, 'rb')
        task_idx = pickle.load(infile)

        if args.pacing_clients > 0:

            if task_no < (args.num_tasks - 1):
                file_name_2 = f'{FCL_dir}/train/' + '/task_' + str(task_no + 1) + '.pkl'
                infile_2 = open(file_name_2, 'rb')
                task_idx_2 = pickle.load(infile_2)

            if task_no < (args.num_tasks - 2):
                file_name_3 = f'{FCL_dir}/train/' + '/task_' + str(task_no + 2) + '.pkl'
                infile_3 = open(file_name_3, 'rb')
                task_idx_3 = pickle.load(infile_3)

        # get test data
        # the test file is already loaded with dataloader
        test_file_dir = f'{FCL_dir}/test/'
        # infile = open(file_name, 'rb')
        # task_idx = pickle.load(infile)

        client_data = []
        label = []

        print('tasks : ', task_no)
        print('Total clients : ', K_total_clients)

        for round_ in range(0, args.num_rounds):

            if args.prompt_pool and args.shared_prompt_pool:
                if task_no > 0:
                    prev_start = (task_no - 1) * args.top_k
                    prev_end = task_no * args.top_k

                    cur_start = prev_end
                    cur_end = (task_no + 1) * args.top_k

                    if (prev_end > args.size) or (cur_end > args.size):
                        pass
                    else:
                        cur_idx = (
                            slice(None), slice(None),
                            slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (
                            slice(None), slice(cur_start, cur_end))
                        prev_idx = (slice(None), slice(None),
                                    slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (
                            slice(None), slice(prev_start, prev_end))

                        with torch.no_grad():
                            if args.distributed:
                                model.module.e_prompt.prompt.grad.zero_()
                                model.module.e_prompt.prompt[cur_idx] = model.module.e_prompt.prompt[prev_idx]
                                optimizer.param_groups[0]['params'] = model.module.parameters()
                            else:
                                # model.e_prompt.prompt.grad.zero_()
                                model.e_prompt.prompt[cur_idx] = model.e_prompt.prompt[prev_idx]
                                optimizer.param_groups[0]['params'] = model.parameters()

            # Transfer previous learned prompt param keys to the new prompt
            if args.prompt_pool and args.shared_prompt_key:
                if task_no > 0:
                    prev_start = (task_no - 1) * args.top_k
                    prev_end = task_no * args.top_k

                    cur_start = prev_end
                    cur_end = (task_no + 1) * args.top_k

                    with torch.no_grad():
                        if args.distributed:
                            model.module.e_prompt.prompt_key.grad.zero_()
                            model.module.e_prompt.prompt_key[cur_idx] = model.module.e_prompt.prompt_key[prev_idx]
                            optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            model.e_prompt.prompt_key.grad.zero_()
                            model.e_prompt.prompt_key[cur_idx] = model.e_prompt.prompt_key[prev_idx]
                            optimizer.param_groups[0]['params'] = model.parameters()

            print('round : ', round_)
            #         L_subset_clients = random.sample(K_total_clients,
            #                                         no_subset_clients)  # select a subset of clients in the current task
            print(K_total_clients)
            L_subset_clients = np.random.choice(range(len(K_total_clients)),
                                                args.no_subset_clients, replace=False
                                                )  # select a subset of clients in the current task
            print(L_subset_clients)
            #     print(len(task_idx)) # number of clients with the data of the current tasks
            #     break
            all_client_weights_ = []
            all_client_losses_ = []
            total_data_samples = 0
            client_data_sample = []

            if args.pacing_clients > 0:
                no_pacing_clients = math.ceil(args.pacing_clients * len(L_subset_clients))

                normal_clients = L_subset_clients[:-(no_pacing_clients * 2)]
                pacing_client_1 = L_subset_clients[-(no_pacing_clients * 2):-no_pacing_clients]
                pacing_client_2 = L_subset_clients[-no_pacing_clients:]

                print(
                    f'{len(L_subset_clients)} {args.pacing_clients} {math.ceil(args.pacing_clients * len(L_subset_clients))} no_pacing_clients:{no_pacing_clients}, normal clients {normal_clients} , pacing_clients: {pacing_client_2, pacing_client_1}')

                for i in normal_clients:
                    total_data_samples += len(task_idx[i])
                    client_data_sample.append(len(task_idx[i]))
                if task_no < (args.num_tasks - 1):
                    for i in pacing_client_1:
                        total_data_samples += len(task_idx_2[i])
                        client_data_sample.append(len(task_idx_2[i]))

                if task_no < (args.num_tasks - 2):
                    for i in pacing_client_2:
                        total_data_samples += len(task_idx_3[i])
                        client_data_sample.append(len(task_idx_3[i]))
            else:
                for i in L_subset_clients:
                    total_data_samples += len(task_idx[i])
                    client_data_sample.append(len(task_idx[i]))

            c_t = 0

            for each_client in L_subset_clients:

                client_data = []  # empty the client dataframe
                print('client number : ', each_client)
                print('each', len(task_idx[each_client]))  # number of samples in each client for task = 0,1,2...
                # client_data_sample = len(task_idx[each_client])
                client_data_idx = task_idx[each_client]
                local_client_model = []
                #             local_client_model = client_models[each_client] # get the client model
                local_client_model = copy.deepcopy(model)

                optimizer = create_optimizer(args, local_client_model)
                if args.sched != 'constant':
                    lr_scheduler, _ = create_scheduler(args, optimizer)
                elif args.sched == 'constant':
                    lr_scheduler = None

                for idx in client_data_idx:  # get the index of the sample
                    client_data.append(train_dataset[idx])  # client_data.append(train_dataset[idx])
                train_dataloader = DataLoader(client_data, batch_size=args.batch_size, shuffle=True)
                #             print(len(next(iter(train_dataloader))[0])) # size of the batch

                # train_local batch
                # local_client_model_weights_, client_loss = updates.local_train(local_client_model, train_dataloader,
                #                                                        mini_batch_size, round_, task_no,
                #                                                        LEARNING_RATE=0.001,
                #                                                        local_epochs=3)  # get the local trained model
                if args.algo == 'fed_avg':
                    local_client_model_weights_, client_loss, trained_local_model = updates.train_client(
                        model=local_client_model,
                        original_model=original_model,
                        train_dataloader=train_dataloader,
                        mini_batch_size=args.batch_size,
                        optimizer=optimizer,
                        class_mask=class_mask,
                        lr_scheduler=lr_scheduler,
                        global_round=round_,
                        task_no=task_no,
                        local_epochs=args.local_epochs,
                        args=args)
                if args.algo == 'fed_prox':
                    local_client_model_weights_, client_loss, trained_local_model = updates.local_fed_prox(
                        model=local_client_model, original_model=original_model, train_dataloader=train_dataloader,
                        mini_batch_size=args.batch_size, global_round=round_, optimizer=optimizer,
                        class_mask=class_mask, task_no=task_no, lr_scheduler=lr_scheduler,
                        local_epochs=args.local_epochs, args=args)

                if args.algo == 'fed_scaffold':
                    local_client_model_weights_, client_loss, trained_local_model, delta_cnt = updates.local_fed_scaffold(
                        model=local_client_model,
                        original_model=original_model,
                        train_dataloader=train_dataloader,
                        mini_batch_size=args.batch_size,
                        optimizer=optimizer,
                        class_mask=class_mask,
                        lr_scheduler=lr_scheduler,
                        global_round=round_,
                        task_no=task_no,
                        local_epochs=args.local_epochs, c_t=c_t,
                        args=args)

                # test_stats = updates.evaluate_till_now_FCL(model=trained_local_model, original_model=original_model,
                #                                            test_file_dir=test_file_dir, device=device,
                #                                            task_no=task_no, acc_matrix=acc_matrix,
                #                                            test_dataset=test_dataset,
                #                                            args=args)

                all_client_weights_.append(local_client_model_weights_)
                all_client_losses_.append(client_loss)
            loss_task_log = f"loss_task_{task_no}"
            wandb.log({loss_task_log: np.mean(all_client_losses_)}, round_)
            #         print(len(client_updated_models_))
            averaged_weights_ = updates.average_weights(all_client_weights_, client_data_sample,
                                                        total_data_samples)  # aggregate the local client models
            # old_weights = model.state_dict()['prompt.prompt_key']
            # weigth_diff =  averaged_weights_['prompt.prompt_key'] - old_weights
            # print(torch.sum(weigth_diff,1))

            model.load_state_dict(averaged_weights_)
            # pickle_model.load_state_dict(averaged_weights_)

            batch_loss, batch_acc1, batch_acc5 = evaluate_model.evaluate_rounds(model=model,
                                                                                test_file_dir=test_file_dir,
                                                                                writer=writer,
                                                                                task_no=task_no,
                                                                                test_dataset=test_dataset,
                                                                                round=round_, args=args)

            writer.add_scalar(tag=f"{task_no}loss", scalar_value=batch_loss, global_step=round_)
            # writer.add_scaler(f"{task_no}/acc1 ", batch_acc1, round_)
            writer.add_scalar(tag=f"{task_no}acc1", scalar_value=batch_acc1, global_step=round_)
            # writer.add_scaler(f"{task_no}/acc_cal", correct, round_)

        # (model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
        # device, task_id=-1, class_mask=None, args=None,):
        # # test on the classes seen untill now
        # load the model for testing

        # test_stats, forgetting = evaluate_model.evaluate_trained_model(model=model,
        #                                             # original_model=original_model,
        #                                            test_file_dir=test_file_dir, device=device,
        #                                            task_no=task_no, acc_matrix=acc_matrix, test_dataset=test_dataset,
        #                                            args=args)

        test_stats, forgetting = evaluate_model.evaluate_trained_model(model=model,
                                                                       test_file_dir=test_file_dir, device=device,
                                                                       task_no=task_no, acc_matrix=acc_matrix,
                                                                       test_dataset=test_dataset,
                                                                       args=args)

        # Additional information

        writer.add_scalar(tag="total_acc", scalar_value=test_stats[0], global_step=task_no)
        writer.add_scalar(tag="total_loss", scalar_value=test_stats[2], global_step=task_no)
        writer.add_scalar(tag="forgetting", scalar_value=forgetting, global_step=task_no)


        PATH = f'{FCL_save_dir}/{run.name}/checkpoint/'
        if not os.path.exists(PATH):
            os.makedirs(PATH)
            print(f'FCL dir created at : {PATH}')
        else:
            print(f'FCL dir exists, overwritting : {PATH}')

        model_path = PATH + f"task{task_no}_model.pt"

        # file_name = PATH + f"task{task_no}_model.pkl"
        # with open(file_name, 'wb') as outfile:
        #     pickle.dump(pickle_model, outfile)
        torch.save({
            'epoch': task_no,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_path)
        # torch.save(model.state_dict(), f"{PATH}/task{task_no}_model.h5")

    # save the final model
    wandb.save(f'{run.name}model.h5')
    #             label.append(train_dataset.targets[idx])
    # #             plt.figure(num=task_no, figsize=(2, 2))
    # #             plt.title(label=train_dataset.classes[label[0]])
    # #             plt.imshow(data[0])
    # #             print(data[0].shape , label)
    #         break
    #     break
    wandb.finish()


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    sys.exit(0)

