export CUDA_VISIBLE_DEVICES=0,1

python ./lra_benchmarks/text_classification/train.py \
--model_dir ./lra_benchmarks/experiments/text_classification/flowformer_final \
--model_type flowformer \
--config ./lra_benchmarks/text_classification/configs/flowformer_base.py \
--data_dir /data/lra/