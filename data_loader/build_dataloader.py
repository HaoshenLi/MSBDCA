import torch
import numpy as np
from torch.utils.data import DataLoader
from data_loader.dataset import *
import os 

def build_dataloaders(opts, splits = ["train", "valid", "test"], printer = print):
    loaders = {}
    diag_classes, diag_labels = None, None
    for i, split in enumerate(splits):
        dataset = PeritoneumDataset_Clinic(data_dir=opts.data_dir,
                                       split_file=os.path.join(opts.split_dir, split+".csv"),
                                       split=split,
                                       N_lesion=opts.N_lesion,
                                       treatment_ids=opts.treatment_ids,
                                       strong_aug=(split in ["train"]),
                                       thres=opts.thres,
                                       printer=printer)
        
        if i == 0:
            diag_classes = dataset.n_classes
            diag_labels = dataset.diag_labels
            
        loader = DataLoader(dataset, 
                            batch_size=opts.batch_size, 
                            shuffle=(split in ["train"]),
                            pin_memory=True, 
                            num_workers=opts.num_workers)
        
        loaders[split] = loader
        
    # compute class-weights for balancing dataset
    if opts.class_weights:
        assert diag_classes is not None and diag_labels is not None
        class_weights = np.histogram(diag_labels, bins=diag_classes)[0]
        class_weights = np.array(class_weights) / sum(class_weights)
        for i in range(diag_classes):
            class_weights[i] = round(np.log(1.0 / class_weights[i]), 5)
    else:
        class_weights = np.ones(diag_classes, dtype=np.float)
    
    return loaders, diag_classes, class_weights

    