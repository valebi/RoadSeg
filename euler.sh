
USER="$(whoami)"
DATA_DIR="/cluster/scratch/${USER}/roadseg"
LOG_DIR="/cluster/scratch/${USER}/roadseg/logs"
OUT_DIR="output"

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy

pip install -q -r requirements.txt

COMMAND="python main.py \
                --experiment_tag="euler" \
                --wandb \
                --device="cuda" \
                --num_workers=16 \
                --log_dir="${LOG_DIR}" \
                --data_dir="${DATA_DIR}" \
                --test_imgs_dir="${DATA_DIR}/ethz-cil-road-segmentation-2023/test/images" \
                --out_dir="${OUT_DIR}" \
                --make_submission \
                --max_per_dataset=1000 \
                --datasets hofmann \
                --smp_backbone="timm-regnety_080" \
                --smp_encoder_init_weights="imagenet" \
                --pretraining_epochs=1 \
                --finetuning_epochs=5 \
                --train_batch_size=64 \
                --val_batch_size=128
"
echo "${COMMAND}"
sbatch \
    --time=48:00:00 \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=20 \
    -J "roadseg-train" \
    --mem-per-cpu=10000 \
    --gres=gpumem:23240m \
    --gpus=1 \
    --mail-type=ALL \
    --mail-user="${USER}@ethz.ch" \
    --output="logs/last_run.txt" \
    --wrap="${COMMAND}"
