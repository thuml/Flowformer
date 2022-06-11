export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

python -m torch.distributed.launch --nproc_per_node=6 --use_env main.py  --output_dir logs/flowformer --batch-size 128 --data-path /data/ImageNet2012/