# Flowformer (ICML 2022)
Flowformer: Linearizing Transformers with Conservation Flows

Transformers have achieved impressive success in various areas. However, the attention mechanism has a quadratic complexity, significantly impeding Transformers from dealing with numerous tokens and scaling up to bigger models. In pursuing the **linear complexity** and **task-universal** foundation model, we propose Flowformer [[paper]](https://arxiv.org/pdf/2202.06258.pdf) with the following merits:

- **Linear complexity** w.r.t sequence length, can handle extermely long sequence (over 4k tokens)
- **Without specific indcitve bias**, purely derived from the flow network theory
- **Task-universal**, showing strong performance in **$\color{red}{\text{Long sequence, Vision, NLP, Time series, RL}}$**.

## Flow-Attention Design

We cast the attention mechanism into flow network, where the information flow is aggregated from the sources (values) to the sinks (results) through the learned flow capacities (attentions).

By conducting the conservation in both source and sink ascpects, we can bring competition into Flow-Attention design to avoid trivial attention in the spirit that "fixed resource will cause competition''.

<p align="center">
<img src=".\pic\Flow-Attention.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 1.</b> Flow-Attention with Competition and Allocation mechanisms.
</p>

## Get Started

1. Please refer to different folders for detailed experiment instructions.

   Note: We have suffered a lot in configuring environments for different tasks. If you also have problems in solving the environment, feel free to contact us.

2. Progress of benchmarks (All the code will be released before 2022.07.30)

- [x] Core code: see `Flow_Attention.py`
- [ ] Long Sequence Modeling in LRA
- [ ] Vision Recognization in ImageNet-1K
- [ ] Language Modeling in WikiText-103
- [ ] Time series classification  in UEA
- [ ] Reinforcement Learning in D4RL

## Main Results

See the [[paper]](https://arxiv.org/pdf/2202.06258.pdf) for detailed results, including **nearly 20 comparing baselines**. 

| Task                                  | Metrics                                          | Flowformer       | Performer    | Reformer     | Vanilla<br>Transformer |
| ------------------------------------- | ------------------------------------------------ | ---------------- | ------------ | ------------ | ---------------------- |
| Long Sequence Modeling <br> (LRA)     | Avg Acc (%) $\uparrow$                           | **56.48**        | 51.41        | 50.67        | OOM                    |
| Vision Recognization<br>(ImageNet-1K) | Top-1 Acc (%) $\uparrow$                         | **80.4**         | 78.1         | 79.6         | 78.7                   |
| Language Modeling<br>(WikiText-103)   | Perplexity $\downarrow$                          | **30.8**         | 37.5         | 33.6         | 33.0                   |
| Time series classification<br>(UEA)   | Avg Acc (%) $\uparrow$                           | **73.0**         | 71.5         | 71.9         | 71.9                   |
| Offline RL<br>(D4RL)                  | Avg Reward $\uparrow$ <br>Avg Deviation $\downarrow$ | 71.1$\pm$**2.1** | 63.8$\pm$7.6 | 58.3$\pm$9.5 | **72.2**$\pm$2.6       |

Vanilla Transformer means Decision Transorfomer in RL.

## Attention Visualziation

<p align="center">
<img src=".\pic\Attention-visualization.png" height = "300" alt="" align=center />
<br><br>
<b>Figure 2.</b> Attention visualization. Flowformer can capture the essential parts successfully.
</p>

## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{wu2022flowformer,
  title={Flowformer: Linearizing Transformers with Conservation Flows},
  author={Haixu Wu and Jialong Wu and Jiehui Xu and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Machine Learning},
  year={2022}
}
```

## Contact

If you have any questions or want to use the code, please contact whx20@mails.tsinghua.edu.cn.
