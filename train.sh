tensorboard --logdir ./runs --port 6006
# baseline
python train_baseline.py \
  -raw \
  -backbone resnet50 \
  -data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master \
  -train_with_s \
  -train_with_t

python evaluator.py \
  --backbone resnet50 \
  --dataset mmwhs \
  --data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master \
  --restore_from weights/best_Base.mmwhs.s0.f0.v0.resnet50.lr0.00025.mmt0.9.raw.bs32.trainWst.mnmx.e125.Scr0.801.pt \
  --normalization minmax \
  --modality mr \
  --phase test \
  --save_pred \
  --raw \
  --hd --asd

python pretrain_RAIN.py -raw -task pretrain_RAIN \
 -data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master

# SLCL
python train_SLCL.py \
  -raw \
  -backbone resnet50 \
  -data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master \
  -train_with_s \
  -train_with_t \

python train_MCCL.py \
  -lr 8e-4 -raw -train_with_s -train_with_t \
  -thd 0.95 \
  -bs 32 \
  -backbone resnet50 \
  -data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master

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