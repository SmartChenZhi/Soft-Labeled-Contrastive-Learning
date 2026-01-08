tensorboard  --port 6006 --logdir 
# baseline
python train_baseline.py \
  -raw -rev\
  -backbone resnet50 \
  -data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master \
  -train_with_s -train_with_t -epochs 300

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
  -multilvl -d_label_smooth 0.1 -d_update_freq 3 -adjust_lr_dis -lr_dis 1e-5 \
  -epochs 2000 \
  -restore_from weights/AdaptSeg.mmwhs.s0.f0.v0.resnet50.lr0.00025.mmt0.9.raw.bs32.lr_dis1e-05.w_dis0.001.dls0.1.duf3.mutlvl.w_d_aux0.0002.wsegaux0.1.pt \
  -restore_d weights/out_dis_AdaptSeg.mmwhs.s0.f0.v0.resnet50.lr0.00025.mmt0.9.raw.bs32.lr_dis1e-05.w_dis0.001.dls0.1.duf3.mutlvl.w_d_aux0.0002.wsegaux0.1.pt \
  -restore_d_aux weights/out_dis1_AdaptSeg.mmwhs.s0.f0.v0.resnet50.lr0.00025.mmt0.9.raw.bs32.lr_dis1e-05.w_dis0.001.dls0.1.duf3.mutlvl.w_d_aux0.0002.wsegaux0.1.pt

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
  -lr 8e-4 -rev -CNR -CNR_w 4e-5 -clda -intra -phead\
  -wtd_ave -part 2 -bs 16\
  -backbone resnet50 \
  -data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA_png/

python train_BCL.py \
  -raw -backbone resnet50 \
  -data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master \
  -train_with_s -train_with_t \
  -bs 8 -eval_bs 8

python train_AdaptEvery.py \
  -backbone resnet50 \
  -raw -data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master \
  -train_with_s -train_with_t
