salloc \
  -c 64 \
  --mem=720G \
  --partition=gpu_h100 \
  --gres=gpu:h100:4 \
  --time=72:00:00

cd /home/stan/ijepa
source .venv-ijepa/bin/activate

python /gpfs/home4/stan/ijepa/scripts/scan_bad_images.py \
  /scratch-shared/stan/ijepa/imagenet-1k-imagefolder/train \
  --workers 40 \
  --report /tmp/ijepa_bad_images.txt


python main_classification.py   --fname /home/stan/ijepa/configs/classification/in1k_linear_probe.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3