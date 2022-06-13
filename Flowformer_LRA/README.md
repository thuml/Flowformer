# Flowformer for Long Sequence Modeling

We follow the official code base of [[LRA]](https://github.com/google-research/long-range-arena) and implement Flowformer in [[JAX]](https://github.com/google/jax).

<p align="center">
<img src="..\pic\LRA_results.png" height = "300" alt="" align=center />
<br><br>
<b>Figure 1.</b> Results on LRA.
</p>


## Get Started

1. Solve the environment by the following command. For more details and the dataset downloading, you can refer to [[LRA]](https://github.com/google-research/long-range-arena).

```shell
pip install -r requirements.txt
```

2. For all five benchmarks, run the following scripts for reproduction:

```shell
# Listops
bash ./lra_benchmarks/listops_scripts/flowformer.sh
# Text
bash ./lra_benchmarks/text_classification_scripts/flowformer.sh
# Retrieval
bash ./lra_benchmarks/matching_scripts/flowformer.sh
# Image
bash ./lra_benchmarks/image_scripts/flowformer.sh
# Pathfinder
bash ./lra_benchmarks/path_scripts/flowformer.sh
```

## Acknowledgement

We code base is built upon on the official code of LRA:

https://github.com/google-research/long-range-arena

