N_lesion=5
split_dir="../data/internal/split"
in_channels=1
lr=5e-5
batch_size=4
num_workers=8
data_dir="../data/internal/data"
model_name="Model_MSBDCA_Image_Clinic"
save_folder=""
checkpoint=""

CUDA_VISIBLE_DEVICES=0 python main_test_3D.py --data_dir ${data_dir} --split_dir ${split_dir} --in_channels ${in_channels} --checkpoint ${checkpoint} --N_lesion ${N_lesion} --model_name ${model_name} --batch_size ${batch_size} --num_workers ${num_workers} --save_folder ${save_folder}




