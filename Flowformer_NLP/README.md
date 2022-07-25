# Flowformer for Language Modeling

We follow the official code base of [[fairseq]](https://github.com/facebookresearch/fairseq) and implement Flowformer upon that repo. 

Since fairseq a quite large code base, we only provide the changed module and our experimental configuration. You can incorporate `flow_attention.py` to fairseq for reproduction.

<p align="center">
<img src="..\pic\NLP_results.png" height = "300" alt="" align=center />
<br><br>
<b>Figure 1.</b> Results on Wikitext-103.
</p>

## Get Started

1. Solve the environment and download the dataset follows the tutorial of [[Language Modeling]](https://github.com/facebookresearch/fairseq/blob/main/examples/language_model/README.md).
1. Replace the `./fairseq/modules/multihead_attention.py` by our provided `flow_attention.py`.
1. Train and evaluate the model by the following scripts. You can get the pretrained model from [[here]](https://cloud.tsinghua.edu.cn/d/82d93375f97f4ca58886/).

```shell
fairseq-train --task language_modeling \
  data-bin/wikitext-103 \
  --save-dir checkpoints/flowformer \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 6000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 2048 --update-freq 16 \
  --max-update 150000

fairseq-eval-lm data-bin/wikitext-103 \
    --path checkpoints/flowformer/checkpoint_best.pt \
    --batch-size 2 \
    --tokens-per-sample 512 \
    --context-window 400
```

## Acknowledgement

We code base is built upon on the official code of fairseq:

https://github.com/facebookresearch/fairseq

