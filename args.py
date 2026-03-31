from argparse import ArgumentParser


def add_experiment_args(parser: ArgumentParser) -> None:
    # Experiment
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--epochs", default=1200, type=int)
    parser.add_argument("--lr_drop", default=1000, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--checkpoint_dir", default="logs/model", type=str)
    parser.add_argument("--pretrain", action="store_true", help="Enable pretraining")
    parser.add_argument("--ema_decay", default=0.99, type=float)
    parser.add_argument("--uda", action="store_true", help="Enable uda")
    parser.add_argument("--backbone", default="resnet50", type=str)

    # Model parameters
    parser.add_argument("--model", default="BayeSeg", required=False)
    parser.add_argument("--dataset_dir", default="Processed_data_nii", type=str)
    parser.add_argument("--dataset", default="Prostate", type=str)
    parser.add_argument("--in_channels", default=1, type=int)

    # loss weight
    parser.add_argument("--ce_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=0, type=float)
    parser.add_argument("--bayes_loss_coef", default=100, type=float)
    parser.add_argument("--recon_loss_coef", default=1, type=float)


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument("--output_dir", default="./logs/model_scaletransform", type=str)
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="device to use for training / testing",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--num_workers", type=int, default=4)


def add_bayes_args(parser: ArgumentParser) -> None:
    # prior hyper-params for appearance mean m
    parser.add_argument("--mu_0", default=0, type=float)
    parser.add_argument("--sigma_0", default=1, type=float)
    # prior hyper-params for appearance std rho
    parser.add_argument("--phi_rho", default=1e-6, type=float)
    parser.add_argument("--gamma_rho", default=2, type=float)
    # Image boundary upsilon
    parser.add_argument("--phi_upsilon", default=1e-8, type=float)
    parser.add_argument("--gamma_upsilon", default=2, type=float)
    # Seg boundary omega
    parser.add_argument("--phi_omega", default=1e-4, type=float)
    parser.add_argument("--gamma_omega", default=2, type=float)
    # Seg category probability pi
    parser.add_argument("--alpha_pi", default=2, type=float)
    parser.add_argument("--beta_pi", default=2, type=float)

    # LRFS: Lipschitz regularization via frequency spectrum
    parser.add_argument("--use_lrfs", action="store_true", help="Enable LRFS regularization from 3740_paper.pdf")
    parser.add_argument("--lrfs_loss_coef", default=1e-2, type=float)
    parser.add_argument("--lrfs_warmup_epochs", default=100, type=int)
    parser.add_argument("--lrfs_nu_mf", default=0.3, type=float)
    parser.add_argument("--lrfs_nu_hf", default=0.7, type=float)
    parser.add_argument("--lrfs_kappa_mf", default=1.0, type=float)
    parser.add_argument("--lrfs_kappa_hf", default=1.0, type=float)

    # FDI4S: front-door intervention for segmentation
    parser.add_argument("--gs_pretrain_epochs", default=500, type=int)
    parser.add_argument("--gs_loss_coef", default=1.0, type=float)
    parser.add_argument("--fdi_base_channels", default=32, type=int)
    parser.add_argument("--fdi_attn_heads", default=4, type=int)
    parser.add_argument("--gs_num_embeddings", default=256, type=int)
    parser.add_argument("--gs_commitment_cost", default=0.25, type=float)