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
from model.model import *
import json
import pdb
import argparse


def train(opts):

    #setup printer
    os.makedirs(opts.save_dir, exist_ok=True)
    logger = build_logging(os.path.join(opts.save_dir, "log.log"),opts.save_dir)
    printer = logger.info
    printer(json.dumps(opts.__dict__, indent=4, sort_keys=True))

    #setup device and seed
    set_seed(seed=opts.seed)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    #setup model
    model = eval(opts.model_name)()
    model = model.to(device = device)
    if opts.load_checkpoint:
        model.load_state_dict(torch.load(opts.load_checkpoint))
    model_save_path = os.path.join(opts.save_dir,'model.pth')

    #set up data_loader
    loaders, diag_classes, class_weights = build_dataloaders(opts = opts, splits = ["train", "valid" ,"test"], printer = printer)
    train_loader = loaders['train']
    val_loader = loaders['valid']
    test_loader = loaders['test']
    class_weights = torch.from_numpy(class_weights)

    #setup loss_fn
    criterion = build_criterion(opts = opts, class_weights = class_weights.float(), printer = printer).to(device)

    #setup optimizer and lr_scheduler
    optimizer = build_optimizer(model = model, opts = opts, printer = printer)
    lr_scheduler = build_lr_scheduler(opts, printer = printer)
    
    train_losses = []
    train_acc = []
    train_auc_list = []
    val_losses = []
    val_acc = []
    val_auc_list = []
    test_losses = []
    test_acc = []
    test_auc_list = []
    best_auc = 0

    for epoch in range(opts.start_epoch, opts.epoches): 
        printer('开始第{}轮训练'.format(epoch + 1))
        train_loss = 0
        label_list = []
        prob_list = []
        pred_list = []

        model.train()
        for i,item in enumerate(train_loader): 
            
            X = item['X'].to(device)
            MASK = item['MASK'].to(device)
            X_new = torch.concat((X, MASK), dim=2)
            M = item['M'].to(device)
            label = item['y'].to(device)
            batch = X.shape[0]
            if opts.in_channels == 1:
                out = model(X,M,item)['pred']
            elif opts.in_channels == 2:
                out = model(X_new, M, item)['pred']
            else:
                raise Exception('in_channels error')
            loss = criterion(out,label,item)["loss"]
           
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() 

            train_loss += (loss.detach().cpu().numpy() * batch)
            _,pred = out.max(1)  #将每行输出的最大值的索引作为类别预测
            pred_list.extend(pred.detach().cpu().numpy())
            prods = F.softmax(out,dim = 1)
            label_list.extend(label.cpu().numpy())
            prob_list.extend(prods[:,1].detach().cpu().numpy())

        keep_ind = np.where(np.array(label_list) != -1)
        pred_list = np.array(pred_list)[keep_ind]
        label_list = np.array(label_list)[keep_ind]
        prob_list = np.array(prob_list)[keep_ind]
        correct_num = np.sum(pred_list == label_list)
        train_num = len(pred_list)
        train_auc = metrics.roc_auc_score(label_list,prob_list)
        printer('epoch:{},Train Loss:{:3f},Train Auc:{:3f},Train Accucacy:{:3f}'.format(epoch+1,train_loss/train_num,train_auc,correct_num/train_num))
        train_losses.append(train_loss/train_num)
        train_acc.append(correct_num/train_num)
        train_auc_list.append(train_auc)

        #更新学习率
        epoch_lr = lr_scheduler.step(epoch)
        optimizer = update_optimizer(optimizer = optimizer, lr_value=epoch_lr)

        model.eval()
        val_loss = 0
        label_list = []
        prob_list = []
        pred_list = []

        for i,item in enumerate(val_loader):
            
            X = item['X'].to(device)
            MASK = item['MASK'].to(device)
            X_new = torch.concat((X, MASK), dim=2)
            M = item['M'].to(device)
            label = item['y'].to(device)
            batch = X.shape[0]

            if opts.in_channels == 1:
                out = model(X,M,item)['pred']
            elif opts.in_channels == 2:
                out = model(X_new, M, item)['pred']
            else:
                raise Exception('in_channels error')
            loss = criterion(out,label,item)["loss"]

            val_loss += (loss.detach().cpu().numpy() * batch)
            _,pred = out.max(1)  #将每行输出的最大值的索引作为类别预测
            pred_list.extend(pred.detach().cpu().numpy())
            prods = F.softmax(out,dim = 1)
            label_list.extend(label.cpu().numpy())
            prob_list.extend(prods[:,1].detach().cpu().numpy())

        keep_ind = np.where(np.array(label_list) != -1)
        pred_list = np.array(pred_list)[keep_ind]
        label_list = np.array(label_list)[keep_ind]
        prob_list = np.array(prob_list)[keep_ind]
        val_correct_num = np.sum(pred_list== label_list)
        val_num = len(pred_list)
        val_auc = metrics.roc_auc_score(label_list,prob_list)
        printer('epoch:{},Val Loss:{:3f},Val Auc:{:3f},Val Accucacy:{:3f}'.format(epoch+1,val_loss/val_num,val_auc,val_correct_num/val_num))
        val_losses.append(val_loss/val_num)
        val_acc.append(val_correct_num/val_num)
        val_auc_list.append(val_auc)

        if val_auc > best_auc:
            torch.save(model.state_dict(),model_save_path) #保存训练好的模型
            best_auc = val_auc
    
    torch.save(model.state_dict(),os.path.join(opts.save_dir,'model_current.pth'))
    plot(train_losses, val_losses, test_losses, len(train_losses), type='loss', save_path=opts.save_dir)
    plot(train_acc, val_acc, test_acc, len(train_acc), type='acc', save_path=opts.save_dir)
    plot(train_auc_list, val_auc_list, test_auc_list, len(train_auc_list), type='auc', save_path=opts.save_dir)


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
    parser.add_argument("--seed", type=int, default=3407, help="seed")

    parser.add_argument("--model_name", type=str, default='Model_MSBDCA_Image_Clinic', help="model_name")
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
    parser.add_argument("--load_checkpoint", type=str, default="", help="load_checkpoint")
    parser.add_argument("--start_epoch", type=int, default=0, help="start_epoch")

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
    train(args)
    



    
    
    
    
    