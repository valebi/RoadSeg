USER="$(whoami)"
module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy

pip install -q -r requirements.txt

COMMAND="python predict.py"

echo "${COMMAND}"
sbatch \
    --time=12:00:00 \
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
