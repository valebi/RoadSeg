"""Script for parsing command line arguments."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="emgrep: Representation learning framework of emp data for hand \
        movement recognition"
    )
    # SETUP
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Seed for deterministic behavior. Runs nondeterministically if -1.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for training.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Enable Kaggle mode.",
    )
    # LOGGING
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to log to.",
    )
    parser.add_argument(
        "--log_to_file",
        action="store_true",
        help="Log file or stdout.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log to wandb.",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="roadseg",
        help="Name of the wandb project.",
    )
    parser.add_argument(
        "--experiment_tag",
        type=str,
        default="roadseg",
        help="Name of the experiment for logs and wandb.",
    )
    # DATA
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing the data.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hofmann"],
        type=str,
        choices=[
            "cil",
            "onepiece-cil",
            "hofmann",
            "maptiler",
            "esri",
            "bing",
            "bing-clean",
            "roadtracing",
            "epfl",
            "google",
        ],
        help="Datasets to use for pretraining.",
    )
    parser.add_argument("--no_pretrain", action="store_true", help="Disable pretraining.")
    parser.add_argument("--no_finetune", action="store_true", help="Disable pretraining.")
    parser.add_argument(
        "--finetuned_weights_dir",
        type=str,
        default="logs",
        help="Where to store the predictions. MUST'T CONTAIN ANY DIGIT!",
    )
    parser.add_argument("--tta", action="store_true", help="Disable pretraining.")
    parser.add_argument("--tta_all_combinations", action="store_true", help="Disable pretraining.")
    parser.add_argument(
        "--onepiece",
        action="store_true",
        help="Whether to finetune on the pieced-together dataset.",
    )
    parser.add_argument("--slim", action="store_true", help="Reduces no. decoder channels.")
    parser.add_argument(
        "--max_per_dataset",
        type=int,
        default=-1,
        help="Maximum number of images to use per dataset. -1 for all images.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for data loading. (torch)",
    )
    parser.add_argument("--no_data_parallel", action="store_true", help="Disable dataparallel.")
    parser.add_argument("--no_aug", action="store_true", help="Disable augmentations.")

    # INFERENCE
    parser.add_argument(
        "--make_submission", action="store_true", help="Generates a submission file when set."
    )
    parser.add_argument(
        "--test_imgs_dir",
        type=str,
        default="data/ethz-cil-road-segmentation-2023/test/images",
        help="Directory containing the test data.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output",
        help="Where to store the predictions. MUST'T CONTAIN ANY DIGIT!",
    )
    parser.add_argument(
        "--submission_file",
        type=str,
        default="submission.csv",
        help="Full filename of submission CSV file.",
    )
    # MODEL
    parser.add_argument(
        "--smp_model",
        type=str,
        default="Unet",
        choices=["Unet", "UnetPlusPlus", "DeepLabV3", "DeepLabV3+"],
        help="Model (/Framework) for pytorch-segmentation-models",
    )
    parser.add_argument(
        "--smp_encoder_init_weights",
        type=str,
        default="",
        help="Whether to use imagenet initialization for pytorch-segmentation-models",
    )
    parser.add_argument(
        "--initial_model",
        type=str,
        default="",
        help="One can load a pretrained model weights to initialize the model. Must have the same architecture as the model specified by --smp_model.",
    )
    parser.add_argument(
        "--model_download_drive_id",
        type=str,
        default="",
        help="One can download the initial_model weights from a public made google drive file.",
    )
    parser.add_argument(
        "--decoder_depth",
        type=int,
        default=4,
        help="Decoder depth for pytorch-segmentation-models. Can only be 4 or 5. Image size must be divisible by 2^decoder_depth.",
    )
    parser.add_argument(
        "--diffusion_timesteps",
        type=int,
        default=100,
        help="Number of timesteps for diffusion models.",
    )
    parser.add_argument(
        "--partial_diffusion",
        action="store_true",
        help="Whether to use partial labels as input to the diffusion process.",
    )
    parser.add_argument(
        "--smp_backbone",
        type=str,
        default="timm-regnety_080",
        # choices=["timm-regnety_080", "dummy-unet", "efficientnet-b5"],
        help="Backbone for pytorch-segmentation-models",
    )
    parser.add_argument(
        "--use_diffusion",
        action="store_true",
        help="Build the smp model within a DiffusionAdapter.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=48,
        help="Batch size for validation.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=400,
        help="Size of input images. Square images are assumed.",
    )
    parser.add_argument(
        "--pretraining_epochs",
        type=int,
        default=10,
        help="Number of epochs for pretraining.",
    )
    parser.add_argument(
        "--pretraining_lr",
        type=float,
        default=2e-3,
        help="Learning rate for pretraining.",
    )
    parser.add_argument(
        "--metrics_to_watch",
        nargs="+",
        type=str,
        default=["iou"],
        choices=["iou", "f1", "precision", "recall", "compf1"],
        help="Metrics to watch during training. Metric to monitor during training is the first argument it will be used to report the best score and selection of the best model.",
    )
    parser.add_argument(
        "--pretraining_loss",
        type=str,
        default="bce",
        choices=[
            "bce",
            "reg_f1",
            "patch_f1",
            "smp_dice",
            "smp_jaccard",
            "smp_lovasz",
            "smp_tversky",
            "smp_soft_ce",
            "patchgan_disc",
            "patchgan_dice"
        ],
        help="Loss to be used for pretraining.",
    )
    parser.add_argument(
        "--finetuning_loss",
        type=str,
        default="reg_f1",
        choices=[
            "bce",
            "reg_f1",
            "patch_f1",
            "smp_dice",
            "smp_jaccard",
            "smp_lovasz",
            "smp_tversky",
            "smp_soft_ce",
            "patchgan_disc",
            "patchgan_dice"
        ],
        help="Loss to be used for finetuning.",
    )
    parser.add_argument(
        "--finetuning_epochs",
        type=int,
        default=50,
        help="Number of epochs for finetuning.",
    )
    parser.add_argument(
        "--finetuning_lr",
        type=float,
        default=2e-3,
        help="Learning rate for finetuning.",
    )
    parser.add_argument(
        "--n_finetuning_folds",
        type=int,
        default=5,
        help="Number of folds for finetuning.",
    )
    parser.add_argument(
        "--only_fold",
        type=int,
        default=-1,
        help="Will skip all other finetuning folds if set != -1",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "plateau", "exponential", "cosine_warm_restarts"],
        help="Learning rate scheduler.",
    )
    parser.add_argument(
        "--scheduler_warmup_iters",
        type=int,
        default="1",
        help="Learning rate scheduler warmup period. Uses a linear warmup fomr 0.01 lr to 1.0 lr.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for scheduler.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-6,
        help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--discriminator_download_drive_id",
        type=str,
        default="",
        help="One can download a discriminator from a public made google drive file.",
    )
    return parser.parse_args()
