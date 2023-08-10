from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
import argparse
import torch
import numpy as np
import yaml
import json
import random
from trainer import Trainer
import wandb
# import SummaryWriter
from torch.utils.tensorboard import SummaryWriter


def create_args():
    
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()

    # Standard Args
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                         help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--log_dir', type=str, default="outputs/out",
                         help="Save experiments results in dir for future plotting!")
    parser.add_argument('--learner_type', type=str, default='default', help="The type (filename) of learner")
    parser.add_argument('--learner_name', type=str, default='NormalNN', help="The class name of learner")
    parser.add_argument('--debug_mode', type=int, default=0, metavar='N',
                        help="activate learner specific settings for debug_mode")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--overwrite', type=int, default=0, metavar='N', help='Train regardless of whether saved model exists')

    # CL Args          
    parser.add_argument('--oracle_flag', default=False, action='store_true', help='Upper bound for oracle')
    parser.add_argument('--upper_bound_flag', default=False, action='store_true', help='Upper bound')
    parser.add_argument('--memory', type=int, default=0, help="size of memory for replay")
    parser.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
    parser.add_argument('--DW', default=False, action='store_true', help='dataset balancing')
    parser.add_argument('--prompt_param', nargs="+", type=int, default=[1, 1, 1],
                         help="e prompt pool size, e prompt length, g prompt length")

    # Arch params
    parser.add_argument('--mu', type=float, default=0.0)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--eps', type=float, default=0.0)

    # Config Arg
    parser.add_argument('--config', type=str, default="configs/config.yaml",
                         help="yaml experiment config input")
    parser.add_argument('--wandbgroup',type=str, default='temp')
    parser.add_argument('--wandb_name',type=str, default=None)

    # fed args
    parser.add_argument('--fedMoon', nargs="+", type=int, default=[2],
                         help="Fed ConMon 2 is the fedmoon version , 1 is the triplet loss + 1 ortho replacement, 3 is considering fedmoon with 1 ortho replacement")


    return parser

def get_args(argv):
    parser=create_args()
    args = parser.parse_args(argv)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config.update(vars(args))
    return argparse.Namespace(**config)

# want to save everything printed to outfile
class Logger(object):
    def __init__(self, name):
        self.terminal = sys.stdout
        self.log = open(name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()

if __name__ == '__main__':
    args = get_args(sys.argv[1:])

    # determinstic backend
    torch.backends.cudnn.deterministic=True

    # run = wandb.init(project="FedCoda",
    #                  sync_tensorboard=True, group= args.wandbgroup)



    # duplicate output stream to output file
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    log_out = args.log_dir + '/output.log'
    sys.stdout = Logger(log_out)

    # save args
    with open(args.log_dir + '/args.yaml', 'w') as yaml_file:
        yaml.dump(vars(args), yaml_file, default_flow_style=False)

    metric_keys = ['acc','aux_task','mem','time','plastic','til','cka']
    save_keys = ['global', 'pt', 'pt-local']
    global_only = ['mem','time','plastic','til','cka']
    avg_metrics = {}
    for mkey in metric_keys: 
        avg_metrics[mkey] = {}
        for skey in save_keys: avg_metrics[mkey][skey] = []

    # wandb.watch(log="all")
    
    start_r = 0   
    for r in range(start_r, args.repeat):

        print('************************************')
        print('* STARTING TRIAL SYNC setting' + str(r+1))
        print('************************************')
        run = wandb.init(project="FedCPrompt_debug",
                     sync_tensorboard=True, group= args.wandbgroup,name = args.wandb_name + str(os.environ.get('RUN_COUNTER', '1')))
        writer = SummaryWriter(log_dir=args.log_dir)
        args.wandb = wandb
        args.writer = writer
        

        # set random seeds
        seed = r
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # set up a trainer
        trainer = Trainer(args, seed, metric_keys, save_keys)

        # init total run metrics storage
        max_task = trainer.max_task
        if r == 0: 
            for mkey in metric_keys: 
                avg_metrics[mkey]['global'] = np.zeros((max_task,args.repeat))
                if (not (mkey in global_only)):
                    avg_metrics[mkey]['pt'] = np.zeros((max_task,max_task,args.repeat))
                    avg_metrics[mkey]['pt-local'] = np.zeros((max_task,max_task,args.repeat))

        # train model
        avg_metrics = trainer.train(avg_metrics)  

        # evaluate model
        # avg_metrics = trainer.evaluate(avg_metrics)    

        # save results
        for mkey in metric_keys: 
            m_dir = args.log_dir+'/results-'+mkey+'/'
            if not os.path.exists(m_dir): os.makedirs(m_dir)
            for skey in save_keys:
                if (not (mkey in global_only)) or (skey == 'global'):
                    save_file = m_dir+skey+'.yaml'
                    result=avg_metrics[mkey][skey]
                    yaml_results = {}
                    if len(result.shape) > 2:
                        yaml_results['mean'] = result[:,:,:r+1].mean(axis=2).tolist()
                        if r>1: yaml_results['std'] = result[:,:,:r+1].std(axis=2).tolist()
                        yaml_results['history'] = result[:,:,:r+1].tolist()
                    else:
                        yaml_results['mean'] = result[:,:r+1].mean(axis=1).tolist()
                        if r>1: yaml_results['std'] = result[:,:r+1].std(axis=1).tolist()
                        yaml_results['history'] = result[:,:r+1].tolist()
                    with open(save_file, 'w') as yaml_file:
                        yaml.dump(yaml_results, yaml_file, default_flow_style=False)

        # Print the summary so far
        print('===Summary of experiment repeats:',r+1,'/',args.repeat,'===')
        for mkey in metric_keys: 
            print(mkey, ' | mean:', avg_metrics[mkey]['global'][-1,:r+1].mean(), 'std:', avg_metrics[mkey]['global'][-1,:r+1].std())
        wandb.finish()