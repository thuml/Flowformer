export CUDA_VISIBLE_DEVICES=6,7

python ./lra_benchmarks/image/train.py \
--task_name cifar10 \
--model_dir ./lra_benchmarks/experiments/image/flowformer_final \
--model_type flowformer \
--config ./lra_benchmarks/image/configs/cifar10/flowformer_base.py
