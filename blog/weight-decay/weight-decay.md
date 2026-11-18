---
title: Weight Decay Under BatchNorm
subtitle: One hyperparameter, three architectures, Oxford Flowers-102. Same penalty, different jobs.
date: 2026-11-18
authors:
  - name: Eleijahmar Fifth C. Sevare
tags: [deep learning, weight decay, regularization, pytorch, flowers102]
---

Weight decay is usually explained as a penalty on large weights that keeps a model simple. That
is exactly what it does in a plain network. In a network built from BatchNorm layers, most
weights can be scaled up or down without changing the output at all, so a penalty on their size
can't simplify anything. Yet weight decay is standard in every ResNet recipe.

> **Question.** On a dataset small enough that every model overfits, how does weight decay change
> accuracy and the size of the weights for an MLP, a small CNN and ResNet-18? Does it do the same
> job in all three?

## What weight decay does

PyTorch's SGD with `weight_decay=λ` adds $\lambda\theta$ to every gradient before the momentum
step. Without momentum, for clarity, the update is

$$
\theta_{t+1} = \theta_t - \eta\big(\nabla\mathcal{L}(\theta_t) + \lambda\,\theta_t\big)
= (1-\eta\lambda)\,\theta_t - \eta\,\nabla\mathcal{L}(\theta_t),
$$

so every step shrinks all weights by a factor $(1-\eta\lambda)$ before following the gradient.
This is the same as training on $\mathcal{L}(\theta) + \tfrac{\lambda}{2}\lVert\theta\rVert^2$,
an L2 penalty. (For SGD the two are identical; for Adam they are not [^loshchilov].) The classic
argument is that smaller weights mean a smoother function that can't fit noise in the training
labels [^krogh].

**Where the models differ.** In ResNet-18 every convolution is followed by BatchNorm, which
divides by the batch's standard deviation. Scaling a convolution's weights $w$ by any $c > 0$
scales its output by $c$, and BatchNorm divides it right back out:

$$
\mathcal{L}(c\,w) = \mathcal{L}(w)
\quad\Longrightarrow\quad
w \cdot \nabla_w\mathcal{L} = 0,
\qquad
\lVert\nabla_w\mathcal{L}(c\,w)\rVert = \tfrac{1}{c}\,\lVert\nabla_w\mathcal{L}(w)\rVert.
$$

So for these weights the penalty can't make the function simpler: the function doesn't depend on
$\lVert w\rVert$. What $\lVert w\rVert$ does control is how far each step *turns* the weights.
The gradient shrinks like $1/\lVert w\rVert$ and the angle a step moves through like
$1/\lVert w\rVert$ again, so the effective learning rate is about $\eta/\lVert w\rVert^2$
[^laarhoven][^zhang]. Without decay, steps that are always perpendicular to $w$ make
$\lVert w\rVert$ grow, and the effective learning rate quietly decays. With decay, the norm
settles at an equilibrium and the effective learning rate stays high.

In ResNet-18, then, weight decay is mostly a **learning-rate control**. Only the final classifier
(and BatchNorm's own scale and shift, which PyTorch also decays by default) are penalised in the
classic sense. In the MLP and CNN, which have no normalisation, weight decay is a true
**complexity penalty** on every layer.

## Experiment setup

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Reserted/Data-Science-Portfolio/blob/main/blog/weight-decay/weight-decay-colab.ipynb)

Every figure and number below comes from
[`weight-decay-colab.ipynb`](weight-decay-colab.ipynb), run on a free Colab T4 GPU.

| Setting | Value |
|---|---|
| Dataset | Oxford Flowers-102 [^flowers] (`torchvision.datasets.Flowers102`): 102 flower species, resized and centre-cropped to 64×64 RGB |
| Split | official: **1,020 train (10 per class)** / 1,020 validation / 6,149 test |
| MLP | 12,288 → 512 → 256 → 102, ReLU; 6,449,510 parameters |
| CNN | two 3×3 conv blocks (32, 64 channels) with max-pool → dense 128 → 102; 2,129,830 parameters |
| ResNet-18 | `torchvision.models.resnet18`, random init (no pretraining); 11,228,838 parameters |
| Optimizer | SGD, momentum 0.9, learning rate 0.01 |
| Held fixed | batch size 32 (31 steps per epoch), no dropout, no augmentation |
| **Swept** | **weight decay λ ∈ {0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2}** |
| Runs | 6 strengths × 3 models × 3 seeds = 54 runs, 50 epochs each |
| Measured | final-epoch validation accuracy, L2 norm of all weights $\lVert\theta\rVert$ per epoch; test accuracy for each model's best λ |

Ten images per class, trained from scratch with no augmentation, is deliberately hard. Absolute
accuracy will be low for every model. What matters here is how it moves with λ. Within a seed,
every λ starts from **identical initial weights**.

```python
df = sweep("weight_decay", [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2], epochs=50, batch_size=32)
```

## Prediction

Written before running the sweep:

1. MLP and CNN follow the textbook shape: a little decay helps validation accuracy, and
   $\lambda = 10^{-2}$ shrinks the weights so hard that the models underfit.
2. Without decay, ResNet-18's weight norm keeps growing through training; with decay it levels
   off, and the stronger the decay the lower the plateau.
3. ResNet-18's accuracy responds to λ through its effective learning rate rather than through
   underfitting, so its curve has a different shape from the other two.

## Results

```{figure} figures/samples.png
:alt: One Flowers-102 training image from each of the first ten classes.

The data: one 64×64 training image from each of the first ten of 102 classes.
```

```{figure} figures/accuracy-vs-weight-decay.png
:alt: Final validation accuracy against weight decay, one line per model, with a band showing the spread across three seeds.
:name: fig-wd-accuracy

Final validation accuracy after 50 epochs against weight decay. Lines are the mean of three
seeds; the band spans the worst and best seed.
```

```{figure} figures/weight-norm-curves.png
:alt: L2 norm of all weights per epoch for each model, one line per weight decay strength.
:name: fig-wd-norm

The L2 norm of all weights during training, one panel per model. Darker lines are stronger
decay. Each panel has its own y-axis; the models have very different parameter counts.
```

```{figure} figures/val-loss-curves.png
:alt: Validation loss per epoch for each model, one line per weight decay strength.
:name: fig-wd-curves

Validation loss per epoch, one panel per model. A curve that turns upward is the model
overfitting.
```

<!-- Paste figures/results.md here, then answer:
     - Do MLP and CNN show a peak, then collapse at 1e-2 (fig-wd-accuracy)? Prediction 1.
     - Does ResNet-18's norm grow at λ=0 and plateau with decay (fig-wd-norm)? Prediction 2.
     - Is ResNet-18's accuracy curve shaped differently? Prediction 3.
     - Does decay stop the late rise in validation loss (fig-wd-curves)? -->

## Takeaway

<!-- One paragraph: is weight decay the same regulariser in all three models? -->

[^loshchilov]: I. Loshchilov, F. Hutter, "Decoupled Weight Decay Regularization," ICLR 2019.
[^krogh]: A. Krogh, J. A. Hertz, "A Simple Weight Decay Can Improve Generalization," NeurIPS
    1991.
[^laarhoven]: T. van Laarhoven, "L2 Regularization versus Batch and Weight Normalization,"
    arXiv:1706.05350, 2017.
[^zhang]: G. Zhang, C. Wang, B. Xu, R. Grosse, "Three Mechanisms of Weight Decay
    Regularization," ICLR 2019.
[^flowers]: M.-E. Nilsback, A. Zisserman, "Automated Flower Classification over a Large Number
    of Classes," ICVGIP 2008.
