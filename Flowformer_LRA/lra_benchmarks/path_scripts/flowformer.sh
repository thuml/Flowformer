export CUDA_VISIBLE_DEVICES=2,3

python ./lra_benchmarks/image/train.py \
--task_name pathfinder32_hard \
--model_dir ./lra_benchmarks/experiments/path/flowformer_final \
--model_type flowformer \
--config ./lra_benchmarks/image/configs/pathfinder32/flowformer_base.py