import random
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging

def set_seed(seed=3407):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def plot(train_list, val_list, test_list, epoches, type, save_path):
	epoches_list = [i + 1 for i in range(epoches)]
	plt.plot(epoches_list,train_list,'r')
	plt.plot(epoches_list,val_list,'b')
	plt.plot(epoches_list,test_list,'g')
	plt.legend(['train','val', 'test'])
	plt.xlabel('epoch')
	assert type == 'loss' or type == 'acc' or type == 'auc','type设置错误'
	if type == 'loss':
		plt.ylabel('loss')
		plt.title('loss')
		plt.savefig(os.path.join(save_path,'loss.png'))
	elif type == 'acc':
		plt.ylabel('acc')
		plt.title('acc')
		plt.savefig(os.path.join(save_path,'acc.png'))
	elif type == 'auc':
		plt.ylabel('auc')
		plt.title('auc')
		plt.savefig(os.path.join(save_path,'auc.png'))
	plt.close()

def build_logging(filename,logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level = logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    return logger
