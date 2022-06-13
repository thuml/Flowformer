export CUDA_VISIBLE_DEVICES=4,5

python ./lra_benchmarks/matching/train.py \
--config ./lra_benchmarks/matching/configs/flowformer_base.py \
--model_dir ./lra_benchmarks/experiments/matching/flowformer_final \
--model_type flowformer_retrieval \
--data_dir /data/lra/lra_release/lra_release/tsv_data/ \
--vocab_file_path /data/lra/aan