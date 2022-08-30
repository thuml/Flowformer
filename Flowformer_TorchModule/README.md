# Flowformer as Pytorch Module

We build a pytorch module for Flowformer upon the [miniGPT](https://github.com/karpathy/minGPT) repo, such that you can `import flowformergpt` into your project by the following instructions:

```
cd Flowformer_TorchModule
pip install -e .
```

## Usage

Here's how you'd instantiate a FlowFormer in the GPT version.

```python
from flowformergpt.model import FlowFormer_GPT

model_config = FlowFormer_GPT.get_default_config()
model_config.model_type = 'flowformer-gpt2'
model_config.vocab_size = 50257  # openai's model vocabulary
model_config.block_size = 1024  # openai's model block_size (i.e. input context length)
model = FlowFormer_GPT(model_config)
```

And here's how you'd train it:

```python
# your subclass of torch.utils.data.Dataset that emits example
# torch LongTensor of lengths up to 1024, with integers from [0,50257)
train_dataset = YourDataset()

from flowformergpt.trainer import Trainer

train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4  # many possible options, see the file
train_config.max_iters = 1000
train_config.batch_size = 32
trainer = Trainer(train_config, model, train_dataset)
trainer.run()
```

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base:

https://github.com/karpathy/minGPT
