tensorboard --logdir ./runs --port 6006
# baseline
CUDA_VISIBLE_DEVICES=1 python /data6/smartchen/code/SLCL/train_baseline.py \
  -raw \
  -data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master \
  -train_with_s \
  -train_with_t

# SLCL
python train_SLCL.py \
  -raw \
  -backbone resnet50 \
  -data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master \
  -train_with_s \
  -train_with_t \


CUDA_VISIBLE_DEVICES=1 python /data6/smartchen/code/SLCL/train_BCL.py \
  -raw -backbone resnet50 \
  -data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master \
  -train_with_s -train_with_t \
  -bs 8 -eval_bs 8

# SLCL
# in /data6/smartchen/code/SLCL/train.sh
CUDA_VISIBLE_DEVICES=0 python /data6/smartchen/code/SLCL/train_AdaptEvery.py \
  -backbone resnet50 \
  -raw -data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master \
  -train_with_s -train_with_t

python /data6/smartchen/code/SLCL/compute_class_centers.py \
  -data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master \
  -raw \
  -domain s \
  -method hard_s \
  -fold 0 \
  -backbone resnet50 \
  -batch_size 8

python /data6/smartchen/code/SLCL/compute_class_centers.py \
  -data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master \
  -raw \
  -domain t \
  -method soft_t \
  -fold 0 \
  -backbone resnet50 \
  -weighted_ave \
  -threshold -2 \
  -low_thd 0.6 \
  -high_thd 0.99 \
  -batch_size 8