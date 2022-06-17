cd data
python download_d4rl_datasets.py
cd .. 

for dataset_type in medium medium-replay medium-expert; do
    for env_name in halfcheetah hopper walker2d; do
        python experiment.py --env $env_name --dataset $dataset_type --only_prepare_dataset
    done
done