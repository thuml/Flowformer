# Flowformer for Time Series Classification

We test our proposed Flowformer on the [[UEA]](https://www.timeseriesclassification.com/) dataset, including 10 subsets.

<p align="center">
<img src="..\pic\TS_results.png" height = "300" alt="" align=center />
<br><br>
<b>Figure 1.</b> Results on UEA.
</p>


## Get Started

1. Install torch, numpy.

2. Download the dataset from [[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/d/20ec246857454850a1f3/).

3. Train and evaluate the model with following commands. We use the "Best accuracy" as our metric for all baselines and experiments.

```shell
bash scripts/flowformer.sh
```

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base:

https://github.com/gzerveas/mvts_transformer

https://github.com/thuml/Autoformer
