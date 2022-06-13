export CUDA_VISIBLE_DEVICES=2,3,4,5

python ./lra_benchmarks/listops/train.py \
--model_dir ./lra_benchmarks/experiments/listops/flowformer_final \
--model_type flowformer \
--config ./lra_benchmarks/listops/configs/flowformer_base.py \
--data_dir /data/lra/listops-1000/