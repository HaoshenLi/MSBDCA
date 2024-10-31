import numpy as np
import torch.nn as nn
import torch
import os
import torch.nn.functional as F
from sklearn import metrics
from data_loader.build_dataloader import *
from criterion.build_criterion import *
from optimizer.build_optimizer import *
from optimizer.lr_scheduler import *
from utils.utils import *
import json
import pdb
import argparse
import pickle
from tqdm import tqdm
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines import KaplanMeierFitter
 

def Find_Optimal_Cutoff(TPR, FPR, threshold):
	
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point
   
def test(opts):

    #setup device and seed
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    #setup model
    model = eval(opts.model_name)()
    model.load_state_dict(torch.load(opts.checkpoint, map_location=torch.device('cpu')))
    model = model.to(device = device)

    #set up data_loader
    loaders, diag_classes, class_weights = build_dataloaders(opts = opts, splits = ["test"], printer = print)
    test_loader = loaders['test']

    model.eval()
    label_list = []
    prob_list = []
    pred_list = []
    pid_list = []
    OS_list = []
    OSCensor_list = []
    os.makedirs(opts.save_folder, exist_ok=True)
    save_predictions_path = os.path.join(opts.save_folder, 'predictions.txt')
    save_KM_path = os.path.join(opts.save_folder, 'KM_new.png')

    for i,item in tqdm(enumerate(test_loader)):
        
        pid = item['pid']
        OS = (item['OS'] / 30).tolist()
        OSCensor = item['OSCensor'].tolist()
        X = item['X'].to(device)
        MASK = item['MASK'].to(device)
        X_new = torch.concat((X, MASK), dim=2)
        M = item['M'].to(device)
        label = item['y'].to(device)

        if opts.in_channels == 1:
            out = model(X, M, item)['pred']
        elif opts.in_channels == 2:
            out = model(X_new, M, item)['pred']
        else:
            raise Exception('in_channels error')

        prods = F.softmax(out, dim=1)
        pred = torch.where(prods[:, 1] > 0.3930, 1, 0)
        pred_list.extend(pred.detach().cpu().numpy())
        label_list.extend(label.cpu().numpy())
        prob_list.extend(prods[:, 1].detach().cpu().numpy())
        pid_list.extend(pid)
        OS_list.extend(OS)
        OSCensor_list.extend(OSCensor)

    df = pd.DataFrame({'pid': pid_list, 'label': label_list, 'prob': prob_list, 'OS': OS_list, 'OSCensor': OSCensor_list})
    df.to_csv(save_predictions_path, index=None)

    #计算C_index
    c_index = concordance_index(OS_list, prob_list, OSCensor_list)
    print('c_index:', c_index)

    #画KM曲线
    survival_mean = np.median(np.array(prob_list))
    groups = []
    times = []
    events = []
    group1_time = [] #低风险
    group1_event = [] 
    group2_time = [] ##高风险
    group2_event = []

    for i, idx in enumerate(pid_list):
        survival = prob_list[i]
        OS = OS_list[i]
        OScensor = OSCensor_list[i]
        if survival >= survival_mean:
            group1_time.append(OS)
            group1_event.append(OScensor)
            groups.append(1)
        else:
            group2_time.append(OS)
            group2_event.append(OScensor)
            groups.append(0)
        times.append(OS)
        events.append(OScensor)

    p_value = multivariate_logrank_test(times, groups, events).p_value
    print('p_value:', p_value)
    
    kmf = KaplanMeierFitter()
    kmf.fit(group1_time, group1_event, label='low risk')
    kmf.plot(ci_show = False, label='low risk', color='blue')
    kmf.fit(group2_time, group2_event, label='high risk')
    kmf.plot(ci_show = False, label='high risk', color='red')

    plt.xlabel('Survival(Months)', fontsize=12)
    plt.ylabel('Comulative Survival', fontsize=12)
    plt.text(x=30, y=0.75, s='p_value:{:.4f}'.format(p_value), fontdict={'size': 12})
    plt.legend(fontsize=12)
    plt.title('MSBDCA(Image)')
    plt.show()
    plt.savefig(save_KM_path)
    plt.close()

    #计算AUC
    keep_ind = np.where(np.array(label_list) != -1)
    pred_list = np.array(pred_list)[keep_ind]
    label_list = np.array(label_list)[keep_ind]
    prob_list = np.array(prob_list)[keep_ind]
    pid_list = np.array(pid_list)[keep_ind]
    test_auc = metrics.roc_auc_score(label_list, prob_list)
    print('AUC:{:.4f}'.format(test_auc))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Peritneum Prognosis')
   
    parser.add_argument('--data_dir', type=str, default='../data/internal/data', help='root directory of the dataset')
    parser.add_argument('--split_dir', type=str, default='../data/internal/split', help='split_dir')
    parser.add_argument('--N_lesion', type=int, default=5, help='N_lesion')
    parser.add_argument('--treatment_ids', type=list, default=[0, 1], help='treatment_ids')
    parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers")
    parser.add_argument("--class_weights", type=bool, default=True, help="class_weights")
    parser.add_argument("--epoches", type=int, default=1000, help="epoches")

    parser.add_argument("--model_name", type=str, default='Model_MSBDCA_Image_Clinic', help="model_name")
    parser.add_argument("--checkpoint", type=str, default='', help="checkpoint")
    parser.add_argument("--pretrained_or_not", type=bool, default=False, help="pretrained_or_not")
    parser.add_argument("--linear_features", type=int, default=128, help="projection_features")
    parser.add_argument("--num_head", type=int, default=1, help="num_head")
    parser.add_argument("--num_layer", type=int, default=1, help="num_layer")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--in_channels", type=int, default=1, help="in_channels")
    parser.add_argument("--thres", type=int, default=365, help="thres")
    parser.add_argument("--PCI_fusion", type=int, default=1, help="PCI_fusion")
    parser.add_argument("--PCI_cls_num", type=int, default=3, help="PCI_cls_num")
    parser.add_argument("--vol_type", type=str, default="gtv_volume", help="vol_type")
    parser.add_argument("--pooling", type=str, default="mean", help="pooling")
    parser.add_argument("--save_folder", type=str, default="", help="save_folder")

    parser.add_argument("--optim", type=str, default='sgd', help="optim")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam_beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam_beta2")
    parser.add_argument("--scheduler", type=str, default='fixed', help="scheduler")
    parser.add_argument("--lr", type=float, default=5e-5, help="lr")
    parser.add_argument("--weight_decay", type=float, default=4e-6, help="weight_decay")

    parser.add_argument("--loss_fn", type=str, default='ce', help="loss_fn")
    parser.add_argument("--ce_weight", type=int, default=1, help="ce_weight")
    parser.add_argument("--ds_weight", type=int, default=1, help="ds_weight")

    parser.add_argument("--save_dir", type=str, default='', help="save_dir")
    args = parser.parse_args()
    test(args)
    



    
    
    
    
    