from model.vgg_3d import *
from model.model import *
import torch
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import random
from lifelines.utils import concordance_index
from sklearn import metrics
import matplotlib.pyplot as plt
from lifelines.statistics import multivariate_logrank_test
from lifelines import KaplanMeierFitter


def check_data_info():
    #检查内部数据的平均大小信息和spacing信息
    idx_list = pd.read_csv('../data/internal/split/all.csv')['pid'].to_list()
    folder = '../data/internal/data'
    shape_list = []
    spacing_list = []
    for idx in tqdm(idx_list):
        mask_path = os.path.join(folder, str(idx), 'mask.nii.gz')
        mask = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask)
        shape_list.append(list(mask_array.shape))
        spacing_list.append(list(mask.GetSpacing()))
    shape_array = np.array(shape_list)
    spacing_array = np.array(spacing_list)
    print()


def calculate_bootstrap():
    #随机采样500次，计算这500次结果的平均值和方差
    result_path = ''
    test_num = 30 
    with open(result_path, 'r') as file:
        lines = file.readlines()[1:]
    C_index_list = []
    AUC_list = []
    for i in tqdm(range(5000)):
        selected_lines = [random.choice(lines) for _ in range(test_num)]
        label_list = []
        prob_list = []
        OS_list = []
        OSCensor_list = []
        for line in selected_lines:
            data_items = line.strip().split(',')  # 假设数据项是由逗号分隔的
            label_list.append(int(data_items[1]))
            prob_list.append(float(data_items[2]))
            OS_list.append(float(data_items[3]))
            OSCensor_list.append(int(data_items[4]))
        c_index = concordance_index(OS_list, prob_list, OSCensor_list)
        C_index_list.append(c_index)
        keep_ind = np.where(np.array(label_list) != -1)
        label_list = np.array(label_list)[keep_ind]
        prob_list = np.array(prob_list)[keep_ind]
        test_auc = metrics.roc_auc_score(label_list, prob_list)
        AUC_list.append(test_auc)
    C_index_mean = np.mean(C_index_list)
    C_index_std = np.std(C_index_list)
    C_index_lower_bound = np.percentile(C_index_list, 2.5)
    C_index_upper_bound = np.percentile(C_index_list, 97.5)
    AUC_mean = np.mean(AUC_list)
    AUC_std = np.std(AUC_list)
    AUC_lower_bound = np.percentile(AUC_list, 2.5)
    AUC_upper_bound = np.percentile(AUC_list, 97.5)
    print('C_index mean:{}, std:{}, 95CI:{}-{}'.format(C_index_mean, C_index_std, C_index_lower_bound, C_index_upper_bound))
    print('AUC mean:{}, std:{}, 95CI:{}-{}'.format(AUC_mean, AUC_std, AUC_lower_bound, AUC_upper_bound))


def plot_KM():
    #计算C_index
    result_path = ''
    with open(result_path, 'r') as file:
        lines = file.readlines()[1:]
    label_list = []
    prob_list = []
    OS_list = []
    OSCensor_list = []
    pid_list = []
    for line in lines:
        data_items = line.strip().split(',')  # 假设数据项是由逗号分隔的
        pid_list.append(data_items[0])
        label_list.append(int(data_items[1]))
        prob_list.append(float(data_items[2]))
        OS_list.append(float(data_items[3]))
        OSCensor_list.append(int(data_items[4]))
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
    plt.savefig('')
    plt.close()

    #计算AUC
    keep_ind = np.where(np.array(label_list) != -1)
    label_list = np.array(label_list)[keep_ind]
    prob_list = np.array(prob_list)[keep_ind]
    pid_list = np.array(pid_list)[keep_ind]
    test_auc = metrics.roc_auc_score(label_list, prob_list)
    print('AUC:{:.4f}'.format(test_auc))


  