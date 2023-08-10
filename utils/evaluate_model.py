import numpy as np
import pickle
from torch.utils.data import DataLoader
import torch
from timm.utils import accuracy
from utils import util as utils
from torch.nn.functional import one_hot

def evaluate_trained_model(model, test_dataset,valid_out_dim,  task_no=-1, acc_matrix=None, gpu= None ,num_tasks = 10 ,  wandb = None):
    stat_matrix = np.zeros((3, num_tasks))  # 3 for acc  acc@5 loss
    for i in range(task_no + 1):
        test_dataset.load_dataset(i, train=True)
        test_dataloader = DataLoader(test_dataset, batch_size=128 , shuffle=False)
        test_stats = evaluate(model=model, data_loader=test_dataloader,
                             task_id=i, valid_out_dim = valid_out_dim,  gpu = gpu)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_no] = test_stats['Acc@1']

        # args.wandb.log(
        #     {f"Test/{i}/test_accuracy_{i}": test_stats['Acc@1'], f"Test/{i}/test_Loss_task_{i}": test_stats['Loss']})

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_no + 1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_no + 1,
                                                                                                     avg_stat[0],
                                                                                                     avg_stat[1],
                                                                                                     avg_stat[2])
    forgetting = 0
    if task_no > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                              acc_matrix[:, task_no])[:task_no])
        backward = np.mean((acc_matrix[:, task_no] - diagonal)[:task_no])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)    
    if wandb: 
        wandb.log({f"Test/total/Total_accuracy": avg_stat[0],"global_step":task_no})
        wandb.log({f"Test/total/Total_Loss":  avg_stat[2],"global_step" : task_no})
        wandb.log({f"Test/total/forgetting":  forgetting,"global_step" : task_no})
        print(f"round : {round} : Test/total/Total_accuracy_: {avg_stat[0]}  || Test/total/Total_Loss: {avg_stat[2]} || Test/total/forgetting {forgetting} " )
    else : 
        print(f"round : {round} : Test/total/Total_accuracy_: {avg_stat[0]}  || Test/total/Total_Loss: {avg_stat[2]} || Test/total/forgetting {forgetting} " )

    print(result_str)
    return  avg_stat, forgetting


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader, task_id=-1, class_mask=None, valid_out_dim = None, gpu = False):
    criterion = torch.nn.CrossEntropyLoss()
    # device = torch.device(args.device)
    # criterion = torch.nn.BCELoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(data_loader):
            if gpu:
                images = images.cuda()
                labels = labels.cuda()
                model = model.cuda()



            # compute output
            output = model.forward(images)
            # oo  = model(images) # it is the same as above
            # print(f"output  : {output} \n\n oo : {oo}")
            logits = output[:, :valid_out_dim]
            # print(output.shape , logits.shape)
            # print(f"logits {logits}")

            # preds = torch.sigmoid(logits)
            # labels_oh = one_hot(labels, num_classes=args.no_total_classes)
            loss = criterion(logits, labels)
            # criterion(outputs, labels.float())

            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=images.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=images.shape[0])

    #
    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'],
                  losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




def evaluate_rounds(model: torch.nn.Module,test_dataloader,  task_no: int= -1,
                    test_dataset= None, valid_out_dim = None, 
                     round = None, gpu= None, wandb = None, opname = None, writer = None):


    """ Evaluate local training after each round on aggregated model
    args:
        model: NN to evaulate
        task_no : the current task no
        writer : tensorboard writer
        test_dataset : the test_dataset of the current task
    """
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # criterion = torch.nn.CrossEntropyLoss()

    if opname:
        if task_no > -1:
            test_stats = evaluate(model,test_dataloader, task_id= task_no,valid_out_dim = valid_out_dim, gpu = gpu )
            
            writer.add_scalar(f" {opname}/{task_no}/test_accuracy", test_stats['Acc@1'], round)
            writer.add_scalar(f" {opname}/{task_no}/test_Loss_task", test_stats['Loss'], round)
            wandb.log({f" {opname}/{task_no}/test_Loss_task": test_stats['Loss'], "global_step": round})
            wandb.log({f" {opname}/{task_no}/test_accuracy": test_stats['Acc@1'], "global_step": round})

            # if wandb: 
            #     wandb.log({f" {opname}/{task_no}/test_accuracy": test_stats['Acc@1'], f" {opname}/{task_no}/test_Loss_task": test_stats['Loss']}, step=round)
            #     print(f"round : {round} : {opname}/{task_no}/test_accuracy: {test_stats['Acc@1'] }    {opname}/{task_no}/test_Loss_task: {test_stats['Loss']} ")
            # else : 
            #     print(f"round : {round} :  {opname}/{task_no}/test_accuracy: {test_stats['Acc@1'] }    {opname}/{task_no}/test_Loss_task: {test_stats['Loss']} ")
    else:
        if task_no > -1:
            test_stats = evaluate(model,test_dataloader, task_id= task_no,valid_out_dim = valid_out_dim, gpu = gpu )

            writer.add_scalar(f"Test/{task_no}/test_accuracy", test_stats['Acc@1'], round)
            writer.add_scalar(f"Test/{task_no}/test_Loss_task", test_stats['Loss'], round)
            wandb.log({f"task_{task_no}/val_loss": test_stats['Loss'], "global_step": round})
            wandb.log({f"task_{task_no}/val_acc": test_stats['Acc@1'], "global_step": round})

            # if wandb: 
            #     wandb.log({f"Test/{task_no}/test_accuracy": test_stats['Acc@1'], f"Test/{task_no}/test_Loss_task": test_stats['Loss']}, step=round)
            #     print(f"round : {round} : Test/{task_no}/test_accuracy: {test_stats['Acc@1'] }   Test/{task_no}/test_Loss_task: {test_stats['Loss']} ")
            # else : 
            #     print(f"round : {round} : Test/{task_no}/test_accuracy: {test_stats['Acc@1'] }   Test/{task_no}/test_Loss_task: {test_stats['Loss']} ")
        

    return test_stats['Loss'], test_stats['Acc@1'], test_stats['Acc@5']


