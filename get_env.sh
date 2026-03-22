

# 1. get tiny-imagenet-200 dataset
cd /fnwi_fs/ivi/irlab/datasets/shaomu/ijepa
mkdir -p tiny-imagenet-200

wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip

# 2. get UV venv
cd /fnwi_fs/ivi/irlab/datasets/shaomu/ijepa
source .venv-ijepa/bin/activate

uv pip install \
  torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121

uv pip install \
  pyyaml==6.0.2 \
  numpy==1.26.4 \
  pillow==10.4.0 \
  opencv-python==4.10.0.84 \
  submitit==1.5.2


uv pip install datasets

uv pip install tensorboard
uv pip install setuptools

uv pip install rich


salloc \
  -c 40 \
  --mem=230G \
  --partition=gpu \
  --gres=gpu:4 \
  --nodelist=ilps-cn106 \
  --time=72:00:00

salloc \
  -c 100 \
  --mem=256G \
  --partition=gpu \
  --gres=gpu:8 \
  --nodelist=ilps-cn117,ilps-cn119,ilps-cn120 \
  --time=100:00:00


salloc \
  -c 100 \
  --mem=310G \
  --partition=gpu \
  --gres=gpu:8 \
  --nodelist=ilps-cn117 \
  --time=100:00:00


srun --pty bash

cd /fnwi_fs/ivi/irlab/datasets/shaomu/ijepa
source .venv-ijepa/bin/activate
nvidia-smi

python - <<'PY'
import torch, torchvision, yaml, numpy, PIL, submitit
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("yaml:", yaml.__version__)
print("numpy:", numpy.__version__)
print("PIL:", PIL.__version__)
print("submitit:", submitit.__version__)
print("cuda available:", torch.cuda.is_available())
PY

# 3. 


cd /fnwi_fs/ivi/irlab/datasets/shaomu/ijepa
python main.py \
  --fname toy/tiny_imagenet200_vitt16_ep10.yaml \
  --devices cuda:0 2>&1 | tee toy/tiny_imagenet200_vitt16_ep10.run.log

tensorboard --logdir /fnwi_fs/ivi/irlab/datasets/shaomu/ijepa/toy/tiny_imagenet200_vitt16_ep10_lbs_logs --port 6005

# 4. 

cd /fnwi_fs/ivi/irlab/datasets/shaomu
huggingface-cli download ILSVRC/imagenet-1k --repo-type dataset --local-dir imagenet-1k

cd /fnwi_fs/ivi/irlab/datasets/shaomu
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download ILSVRC/imagenet-1k \
  --repo-type dataset \
  --local-dir imagenet-1k \
  --max-workers 40

python imagenet-1k/export_hf_imagenet_to_imagefolder.py \
  --dataset-path /fnwi_fs/ivi/irlab/datasets/shaomu/imagenet-1k \
  --output-root /fnwi_fs/ivi/irlab/datasets/shaomu/imagenet-1k-imagefolder \
  --split all \
  --num-workers 32


source ~/.bashrc
cd /fnwi_fs/ivi/irlab/datasets/shaomu/ijepa
source .venv-ijepa/bin/activate
export PATH="$HOME/.local/bin:$PATH"
hash -r
which uv

cd /fnwi_fs/ivi/irlab/datasets/shaomu/ijepa

export HF_HOME=/fnwi_fs/ivi/irlab/datasets/shaomu/hf_cache
export HF_DATASETS_CACHE=/fnwi_fs/ivi/irlab/datasets/shaomu/hf_cache/datasets
export HF_HUB_CACHE=/fnwi_fs/ivi/irlab/datasets/shaomu/hf_cache/hub
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"


python main.py \
  --fname /fnwi_fs/ivi/irlab/datasets/shaomu/ijepa/imagenet-1k/in1k_vith14_ep300.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7


python main.py \
  --fname /fnwi_fs/ivi/irlab/datasets/shaomu/ijepa/imagenet-1k/in1k_vith14_ep300.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

tensorboard --logdir /fnwi_fs/ivi/irlab/datasets/shaomu/ijepa/imagenet-1k/vith14.224-bs.2048-ep.300 --port 6008
