# Flowformer for Vision Recognization

We present the Flowformer with 19 layers in a four-stage hierarchical structure, where the channels are in $\{96,192,384,768\}$ and the input sequence length for each stage is in $\{3136,784,196,49\}$ correspondingly.

<p align="center">
<img src="..\pic\CV_results.png" height = "300" alt="" align=center />
<br><br>
<b>Figure 1.</b> Results on ImageNet-1K.
</p>

## Get Started

1. Install torch, torchvision, timm.

2. Download and extract ImageNet images from http://image-net.org/. The directory structure should be

```
│ILSVRC2012/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

3. Train and evaluate the model. You can get **pretrained model** from [[here]](todo).

```shell
bash ./script/flowformer.sh
```

4. For evaluation, please use the argument `--eval`.

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base:

https://github.com/facebookresearch/deit

https://github.com/raoyongming/GFNet
