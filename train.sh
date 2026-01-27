tensorboard  --port 6006 --logdir 
# baseline
python train_baseline.py \
  -raw -rev\
  -backbone resnet50 \
  -data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master \
  -train_with_s -train_with_t -epochs 300

python evaluator.py \
  --backbone drunet \
  --dataset mmwhs \
  --data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master \
  --restore_from weights/best_MPSCL.mmwhs.s0.f0.v0.drunet.32.nb4.bd4.lr0.00025.mmt0.9.raw.bs32.lr_dis0.0001.w_dis0.001.w_mpscl_s1.t0.1.e379.Scr0.683.pt \
  --normalization minmax \
  --modality mr \
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

python pretrain_RAIN.py -raw -rev -task pretrain_RAIN -restore -epochs 2000 -save_every_epochs 200\
 -data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master

# SLCL
python train_SLCL.py \
  -raw -rev -epochs 400 -adjust_lr\
  -backbone drunet \
  -data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master

python train_MCCL.py \
  -lr 8e-4 -rev -raw -CNR -CNR_w 4e-5 -clda -intra -phead -rain\
  -wtd_ave -part 2 -bs 16 -epochs 400 -warmup_epochs 30\
  -backbone drunet -thd 0.95 -seg_pseudo -clbg\
  -data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master/

python train_AdaptEvery.py \
  -backbone resnet50 \
  -raw -data_dir ../data/mmwhs/CT_MR_2D_Dataset_DA-master \
  -train_with_s -train_with_t


tensorboard  --port 6006 --logdir logs/model_unet
python train.py --model BayeSeg --output_dir logs/model_unet --backbone unet
python Trainer_udaBayeSeg.py --model udaBayeSeg --output_dir logs/udaBayeSeg --uda --dataset_dir /root/SLCL/Processed_data_nii_uda
python test.py --model BayeSeg --checkpoint_dir logs/model_unet/best_checkpoint.pth --backbone unet

python train_MPSCL.py -data_dir /root/SLCL/Processed_data_nii_uda\
  -uda -backbone drunet -epochs 2000  -normalization zscore \
  -lr 3e-4 -lr_decay_method linear -lr_decay 1e-3 \
  -adjust_lr_dis -lr_dis 1e-4
