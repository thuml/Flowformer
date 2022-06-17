# Flowformer for Offline RL

## Get Started

### Environment

1. Install [MuJoCo version 2.0](https://www.roboti.us/download.html) at `~/.mujoco/mujoco200` and copy license key to `~/.mujoco/mjkey.txt`

2. Create a conda environment
```
conda env create -f conda_env.yml
conda activate decision-flowformer-gym
```

3. Install [D4RL](https://github.com/rail-berkeley/d4rl)

### Train and evaluate 

1. Download and preprocess datasets with following commands:
```
bash ./script/preprocess_data.sh
```

2. Train and evaluate the model with following commands:
```
bash ./script/flowformer.sh
```

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base:

https://github.com/kzl/decision-transformer