USER="bieriv"
DATA="/cluster/scratch/bieriv/cil"
# @TODO update

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy

pip install -q -r requirements.txt

echo "python main.py --data ${DATA}/data/01_raw --device cuda --debug --wandb --log_dir ${DATA}/logs --lr_cpc ${LR_CPC} --test_idx ${TEST_IDX} --val_idx ${VAL_IDX} --positive_mode ${POSITIVE_MODE} --encoder_dim ${ENC_DIM} --ar_dim ${AR_DIM} --ar_layers ${AR_LAYERS} --split_mode ${SPLIT_MODE} --preprocessing ${PREPROCESSING} --normalize ${NORM} --cpc_alpha ${ALPHA} --weight_decay_cpc ${DECAY} --optimizer_cpc ${OPTIMIZER} --seq_len ${SEQ_LEN} --seq_stride ${SEQ_STRIDE}  --encoder_type ${ENCODER_TYPE} --ar_model ${AR_MODEL} --classifier_type ${CLASSIFIER_TYPE} --debug
sbatch \
    --time=06:00:00 \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=20 \
    -J "emgrep-split-${SPLIT}" \
    --mem-per-cpu=10000 \
    --gres=gpumem:14240m \
    --gpus=1 \
    --mail-type=ALL \
    --mail-user="${USER}@ethz.ch" \
    --output="logs/{}.txt" \
    --wrap="python main.py --data ${DATA} --device cuda --debug --wandb " \
done
