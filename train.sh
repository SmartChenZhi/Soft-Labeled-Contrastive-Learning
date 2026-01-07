tensorboard  --port 6006 --logdir 
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
  --modality ct \
  --phase test \
  --raw

python train_Advent.py \
  -raw \
  -backbone resnet50 \
  -data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master \
  -ent_min -cls_prior

python train_AdaptSeg.py \
  -raw -rev\
  -backbone resnet50 \
  -data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master \
  -multilvl -d_label_smooth 0.1 -d_update_freq 2 -adjust_lr_dis -lr_dis 1e-5

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

python train_BCL.py \
  -raw -backbone resnet50 \
  -data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master \
  -train_with_s -train_with_t \
  -bs 8 -eval_bs 8

python train_AdaptEvery.py \
  -backbone resnet50 \
  -raw -data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master \
  -train_with_s -train_with_t
