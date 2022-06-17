for dataset_type in medium-expert medium medium-replay; do
    for env_name in halfcheetah hopper walker2d; do
        python experiment.py --env $env_name --dataset $dataset_type --work_dir flowformer --n_head 4 --n_layer 3 --embed_dim 256 --learning_rate 0.001
    done
done