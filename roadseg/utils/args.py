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
        default=["cil"],
        type=str,
        choices=["cil", "hofmann", "maptiler", "esri", "bing"],
        help="Datasets to use for pretraining.",
    )
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
    # MODEL
    parser.add_argument(
        "--smp_model",
        type=str,
        default="Unet",
        help="Model (/Framework) for pytorch-segmentation-models",
    )
    parser.add_argument(
        "--smp_backbone",
        type=str,
        default="timm-regnety_080",
        help="Backbone for pytorch-segmentation-models",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Batch size for training. Is multiplied by available #replicas.",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=48,
        help="Batch size for validation. Is multiplied by available #replicas.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=400,
        help="Size of input images. Square images are assumed.",
    )
    parser.add_argument(
        "--pretrain",
        type=bool,
        default=True,
        help="Whether to pretrain the model.",
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
        default=1e-3,
        help="Learning rate for pretraining.",
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
        default=1e-4,
        help="Learning rate for finetuning.",
    )
    parser.add_argument(
        "--n_finetuning_folds",
        type=int,
        default=5,
        help="Number of folds for finetuning.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "plateau", "exponential", "cosine_warm_restarts"],
        help="Learning rate scheduler.",
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

    return parser.parse_args()
