---
title: Learning Rate, Batch Size, Dropout
subtitle: Three hyperparameters, one Fashion-MNIST CNN, 45 seeded training runs, and a test of whether batch size matters on its own
date: 2026-11-18
authors:
  - name: Eleijahmar Fifth C. Sevare
tags: [deep learning, hyperparameters, pytorch lightning, fashion-mnist]
---

Every tutorial hands you `lr=0.01`, `batch_size=64` and `dropout=0.25` as if they were facts of
nature. Change one and the training curve changes shape. But when a run goes badly, which of
the three do you blame? And are they really three separate knobs?

> **Research question.** On a small Fashion-MNIST CNN, how do the learning rate, the batch size
> and the dropout rate each change the training and validation curves? And does batch size
> matter on its own, or only through the learning rate?

These are the first settings anyone touches when a network misbehaves. The learning rate is
often called the single most important hyperparameter [^bengio], batch size decides how fast a
model trains on real hardware [^goyal], and dropout is the default regularizer in most
introductory code [^srivastava]. Knowing what each one does to a learning curve turns tuning
from guessing into diagnosis.

## The three knobs

Training uses **mini-batch stochastic gradient descent (SGD) with momentum**. At each step the
network averages the gradient over a mini-batch $\mathcal{B}_t$ of $B$ examples and folds it into
a running velocity $v$:

$$
v_{t+1} = \mu\, v_t + \frac{1}{B}\sum_{i \in \mathcal{B}_t} \nabla_\theta\, \ell(x_i, y_i; \theta_t),
\qquad
\theta_{t+1} = \theta_t - \eta\, v_{t+1}
$$

Here $\theta$ are the weights, $\ell$ the cross-entropy loss, $\mu = 0.9$ the momentum and $\eta$
the learning rate.

**Learning rate ($\eta$)** is the step size. Too small and the network creeps downhill; too
large and it overshoots and can bounce out of the valley entirely. With momentum 0.9 the
effective step is roughly $\eta/(1-\mu) = 10\eta$.

**Batch size ($B$)** is how many examples are averaged per step. The mini-batch gradient is an
unbiased estimate of the full gradient whose variance shrinks like $1/B$. Small batches mean
noisy steps, and more of them, since one epoch over $N$ images takes $N/B$ steps. Smith and
Le [^smithle] summarise the noise this injects as a *noise scale*

$$
g \;\approx\; \frac{\eta}{1-\mu}\cdot\frac{N}{B},
$$

which depends on $\eta$ and $B$ only through their ratio. That is the idea behind the **linear
scaling rule** [^goyal]: multiply the batch size by $k$, multiply the learning rate by $k$, and
training should behave the same per epoch. If the rule holds, batch size has no effect of its own.

**Dropout ($p$)** switches off each unit with probability $p$ during training and scales the
survivors so the expected activation is unchanged:

$$
\tilde{h}_j = \frac{m_j\, h_j}{1-p}, \qquad m_j \sim \text{Bernoulli}(1-p).
$$

No unit can rely on a particular partner being present, so the network can't memorise through
fragile co-adaptations. It behaves like an ensemble of thinned sub-networks [^srivastava].
Dropout is switched off at evaluation time. That has a side effect: the training accuracy a
progress bar shows is computed *with dropout on*, so it understates how well the network fits
its own training data. I measure it both ways.

All three knobs set **how much noise training sees**: $\eta$ and $B$ through the gradient steps,
$p$ through the activations.

## Experiment setup

The experiment builds on the course notebook `hyperparameters_lightning.ipynb`. It reuses that
notebook's `HParams` config, `FashionMNISTDataModule`, `SmallCNN` and Lightning `LitClassifier`,
and replaces its quick single-seed sweeps with a controlled design. Full code:
[`hyperparameters_lightning.ipynb`](hyperparameters_lightning.ipynb)
([source on GitHub](https://github.com/Reserted/Data-Science-Portfolio/blob/main/blog/archive/lr-batch-size-dropout/hyperparameters_lightning.ipynb)).

| Setting | Value |
|---|---|
| Dataset | Fashion-MNIST [^fashion]: 28×28 grayscale clothing images, 10 classes; no augmentation |
| Split | 60k training images → **54,000 train / 6,000 validation** (same split for every run); 10k test images used once, at the end |
| Model | starter `SmallCNN`: two conv blocks (32, 64 channels, 3×3, max-pool) → dropout → FC 128 → dropout → FC 10; ReLU; BatchNorm **off**; PARAMS parameters |
| Optimizer | SGD, momentum 0.9, no weight decay, constant learning rate |
| **Baseline** | **lr 0.01, batch size 64, dropout 0.25** |
| Learning-rate sweep | lr ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3} |
| Batch-size sweep | B ∈ {16, 64, 256, 1024}, lr fixed at 0.01 |
| Scaled batch-size sweep | B ∈ {16, 64, 256, 1024}, lr = 0.01 × B / 64 |
| Dropout sweep | p ∈ {0, 0.25, 0.5, 0.75} |
| Epochs / seeds | 15 epochs, no early stopping; seeds 0, 1, 2 for every configuration (15 × 3 = 45 runs) |
| Software / hardware | PyTorch 2.7.1, Lightning 2.6.6, `deterministic=True`, NVIDIA GTX 1060 |
| Model selection | highest mean final-epoch **validation** accuracy; test set scored afterwards |

I changed two of the starter's defaults on purpose. **SGD replaces Adam**, because the linear
scaling rule is a statement about SGD and Adam rescales every step on its own. **BatchNorm is
off**, because it normalises with the current batch's statistics, so changing the batch size
would also change the model itself and not just the gradient noise.

Each sweep is a list of `HParams` variants of the baseline:

```python
BASE = replace(HParams(), optimizer="sgd", lr=0.01, momentum=0.9,
               use_batchnorm=False, dropout=0.25, batch_size=64, max_epochs=15)
SWEEPS = {
    "learning_rate": [replace(BASE, lr=v) for v in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]],
    "batch_size":    [replace(BASE, batch_size=b) for b in [16, 64, 256, 1024]],
    "batch_size_scaled_lr": [replace(BASE, batch_size=b, lr=0.01 * b / 64) for b in [16, 64, 256, 1024]],
    "dropout":       [replace(BASE, dropout=p) for p in [0.0, 0.25, 0.5, 0.75]],
}
```

Within a seed, every configuration starts from **identical initial weights**, so differences
come from the hyperparameter rather than a luckier initialisation. Each epoch logs training
metrics (dropout on, averaged during the epoch), validation metrics, and a **train probe**: a
fixed 10,000-image slice of the training set re-scored in eval mode, which makes train and
validation directly comparable.

<!-- RESULTS -->
