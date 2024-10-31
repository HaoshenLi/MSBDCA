#!bin/bash
N_lesion=5
split_dir="../data/internal/split"
in_channels=1
lr=5e-5
thres=365
epoches=1000
model_name="Model_MSBDCA_Image_Clinic"
num_workers=8
save_dir=""

CUDA_VISIBLE_DEVICES=0 python main_train_3D.py --split_dir ${split_dir} --in_channels ${in_channels} --lr ${lr} --save_dir ${save_dir} --N_lesion ${N_lesion} --thres ${thres} --epoches ${epoches} --model_name ${model_name} --num_workers ${num_workers} 
