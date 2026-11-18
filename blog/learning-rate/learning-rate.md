---
title: Learning Rates DO Break
subtitle: Kuzushiji-MNIST.
date: 2026-11-18
authors:
  - name: Eleijahmar Fifth C. Sevare
tags: [deep learning, learning rate, pytorch, kmnist]
---

At a learning rate of 0.3, two of my three networks were dead before their first epoch ended.
They didn't diverge or print NaN. The training loss read 2.308 at epoch 1, and it still read 2.308
at epoch 10. The third network, ResNet-18, trained at the same rate on the same images and
finished at 97.7%.

That number, 2.308, is the best clue in the experiment, and this post works backwards from it.
The setup was simple: an MLP, a small CNN and ResNet-18, six learning rates, three seeds each,
one question.

> **Question.** On one dataset with one optimiser, does the best learning rate move between an
> MLP, a small CNN and ResNet-18? And which architecture tolerates the widest range?

The learning rate is often called the single most important hyperparameter [^bengio], and
tutorials hand out `lr=0.01` as if it were a property of the dataset. It isn't. The step size an
optimiser can afford depends on the shape of the loss surface, and the architecture shapes the
surface, so I expected the best rate to move between models. It didn't: all three peaked at 0.03.
What moved was the cliff one step past the peak. The reason is a property of BatchNorm that lets
ResNet-18 shrink its own step size, and the weight norms in the run logs show it happening.

## The data: cursive from pre-modern books

Kuzushiji-MNIST [^kmnist] has MNIST's shape and a much harder alphabet. Each 28 × 28 image is a
character cut from scanned pages of pre-modern Japanese books, written in *kuzushiji*, a cursive
style that was standard for centuries and that few modern Japanese readers can read. There are ten
classes, one per hiragana, with 6,000 training images each.

```{figure} figures/dataset-kmnist.png
:name: fig-lr-data
:alt: A grid of Kuzushiji-MNIST training images, twelve per class, in ten rows labelled with the modern hiragana and its romanisation (o, ki, su, tsu, na, ha, ma, ya, re, wo). The last column is each class's average image, mostly a shapeless blur.

Twelve random training images per class, with the modern character on the left and the class
average on the right. Rows *ki*, *ha* and *re* each hold shapes that have nothing in common
except the label. Drawn from the raw dataset files by `diagrams/datasets.py`.
```

Look along the *ki*, *ha* and *re* rows of {numref}`fig-lr-data`. Each contains two or more
shapes that share nothing but a label. In kuzushiji one sound could be written with forms
derived from different kanji, so a class here is a *sound*, not a shape. The averages in the
last column make the same point. MNIST's class averages look like blurry digits. Several of
these look like nothing. A model has to learn more than one picture per class, which is why
this dataset is much harder than MNIST at the same size.

## Why there is a ceiling at all

Training uses SGD with momentum, in PyTorch's form. Each step averages the gradient over a
mini-batch $\mathcal{B}_t$, folds it into a velocity $v$, and moves the weights $\theta$ by
$\eta$ times that velocity:

$$
v_{t+1} = \mu\, v_t + \nabla_\theta \mathcal{L}_{\mathcal{B}_t}(\theta_t),
\qquad
\theta_{t+1} = \theta_t - \eta\, v_{t+1},
$$

with momentum $\mu = 0.9$ and learning rate $\eta$.

To see why $\eta$ has a ceiling, look close to a minimum, where the loss is roughly a bowl.
Along the bowl's steepest direction, with curvature $\lambda_{\max}$ (the largest eigenvalue of
the Hessian), momentum SGD converges only when [^goh]

$$
\eta\,\lambda_{\max} < 2(1+\mu).
$$

With $\mu = 0.9$ the ceiling is $\eta < 3.8/\lambda_{\max}$. Below it, a smaller $\eta$ only
means slower progress. Above it, each step overshoots further than the last, and the loss
climbs instead of falling. The ceiling is set by curvature, and curvature is set by the
architecture, which is why I expected the best $\eta$ to move between models.

## Three models, two of them unprotected

The three networks are deliberately ordinary. {numref}`fig-lr-models` shows them as
`make_model()` builds them: a two-hidden-layer MLP, a two-block CNN, and torchvision's
ResNet-18 trained from scratch, with only its first convolution changed to accept one input
channel. In this post every dropout layer has $p = 0$, so the dashed boxes are identity
functions.

```{figure} ../diagrams/models.svg
:name: fig-lr-models
:alt: Three layer stacks side by side. MLP: flatten, linear 512, linear 256, linear n. CNN: two conv-ReLU-maxpool blocks with 32 and 64 channels, then dense 128 and n. ResNet-18: 7x7 stem, max-pool, four stages of two BasicBlocks with 64 to 512 channels, global average pool, linear. The stem and the four stages are shaded to mark BatchNorm.

The three models. Shaded boxes contain BatchNorm. Only ResNet-18 has any. Dashed boxes are
dropout layers, which are identity functions here ($p = 0$). Parameter counts are for $n = 10$
classes.
```

They differ in the two ways that matter most for curvature. **Normalisation:** ResNet-18
follows every convolution with BatchNorm, and the MLP and the CNN have none. **Depth with
shortcuts:** ResNet-18 stacks eight residual blocks, whose identity shortcuts keep gradients
well scaled through the depth. {numref}`fig-lr-resnet` opens one up.

```{figure} ../diagrams/resnet18.svg
:name: fig-lr-resnet
:alt: ResNet-18 stage diagram with spatial sizes per dataset; below, a BasicBlock: conv 3x3, BatchNorm, ReLU, conv 3x3, BatchNorm, addition with the shortcut, ReLU.

ResNet-18 as used here. On 28 × 28 Kuzushiji-MNIST the spatial map is down to 1 × 1 after
`layer4`. Each BasicBlock wraps two convolutions in BatchNorm, the property this post ends up
depending on.
```

BatchNorm normalises each channel by the batch's mean and standard deviation, so the layer's
output does not depend on the scale of the weights feeding it. That is usually explained as
smoothing the loss surface, which lets the network take larger steps [^bjorck][^santurkar].
The results below support a more specific mechanism.

## Keeping the comparison fair

All four posts in this series share one harness, shown in {numref}`fig-lr-harness`. It is built so
that the swept value is the only thing that differs between two runs with the same seed.

```{figure} ../diagrams/harness.svg
:name: fig-lr-harness
:alt: Flow diagram. Data is decoded once, split with seed 42, normalised and kept on the GPU. A sweep loops over value, model and seed; each run seeds, builds the model, then per epoch shuffles, trains with SGD (timed), scores a train probe and the validation set in eval mode, and appends a row. Test accuracy is scored once after the last epoch. Outputs are runs.csv, the figures and results.md.

The experiment harness. The data is decoded, split and normalised once, then stays on the GPU.
Within a seed, every value of the swept hyperparameter starts from the same initial weights and
sees the same batch order.
```

The core of `run()` shows where that control comes from:

```python
torch.manual_seed(seed)   # same seed -> same initial weights for every value of the swept hyperparameter
model = make_model(model_name, x.shape[1], x.shape[2], int(y.max()) + 1, dropout).to(DEVICE)
opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
order_rng = torch.Generator(device=DEVICE).manual_seed(seed)
for epoch in range(1, epochs + 1):
    model.train()
    order = torch.randperm(len(x), device=DEVICE, generator=order_rng)
    for i in range(0, len(x) - batch_size + 1, batch_size):   # drops the last partial batch
        idx = order[i:i + batch_size]
        loss = F.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    train_loss, train_acc = evaluate(model, x[:5000], y[:5000])   # fixed training slice, eval mode
    val_loss, val_acc = evaluate(model, *DATA["val"])
```

Two details matter when reading the figures. The "training loss" plotted below is not the
running average a progress bar shows. It is a fresh evaluation, in eval mode, on a fixed slice of
5,000 training images, so it is directly comparable with validation. And the test set is scored
exactly once per run, after the last epoch. It is never used to choose anything.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Reserted/Data-Science-Portfolio/blob/main/blog/learning-rate/learning-rate-colab.ipynb)

Every figure and number below comes from [`learning-rate-colab.ipynb`](learning-rate-colab.ipynb),
written for a free Colab T4 GPU.

| Setting | Value |
|---|---|
| Dataset | Kuzushiji-MNIST [^kmnist] (`torchvision.datasets.KMNIST`): 28×28 grayscale cursive hiragana, 10 classes |
| Split | 60,000 training images → 54,000 train / 6,000 validation (fixed); official 10,000 test |
| MLP | 784 → 512 → 256 → 10, ReLU; 535,818 parameters |
| CNN | two 3×3 conv blocks (32, 64 channels) with max-pool → dense 128 → 10; 421,642 parameters |
| ResNet-18 | `torchvision.models.resnet18`, random init, first conv takes 1 channel; 11,175,370 parameters |
| Optimiser | SGD, momentum 0.9, constant learning rate |
| Held fixed | batch size 128, no dropout, no weight decay, no augmentation |
| **Swept** | **learning rate ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3}**, half a decade apart |
| Runs | 6 learning rates × 3 models × 3 seeds = 54 runs, 10 epochs each |
| Measured | final-epoch validation accuracy; test accuracy for each model's best rate |

```python
df = sweep("lr", [0.001, 0.003, 0.01, 0.03, 0.1, 0.3], epochs=10)
```

## What I expected

I wrote these down before the first run, so I couldn't quietly revise them afterwards:

1. Every model has a best learning rate somewhere inside the grid: too small is slow, too large
   is unstable.
2. ResNet-18's curve sits further right and is wider. BatchNorm should let it train at 0.1 or
   even 0.3, where the un-normalised MLP and CNN degrade or diverge.
3. At 0.001, ten epochs are not enough for any model to finish learning.

## Results: one optimum, two cliffs

{numref}`fig-lr-accuracy` is the whole experiment in one picture.

```{figure} figures/accuracy-vs-lr.png
:alt: Final validation accuracy against learning rate, one line per model, with a band showing the spread across three seeds. All three rise to a peak at 0.03; at 0.3 the MLP and CNN drop to about 0.1 while ResNet-18 stays near 0.98.
:name: fig-lr-accuracy

Final validation accuracy after 10 epochs. Lines are the mean of three seeds, and bands span the
worst and best seed (mostly thinner than the line). The three curves share a peak at 0.03. The
difference is at 0.3, where the MLP and the CNN fall to chance and ResNet-18 barely moves.
```

| Learning rate | MLP | CNN | ResNet-18 |
|---:|:---:|:---:|:---:|
| 0.001 | 92.3 ± 0.1 | 96.2 ± 0.3 | 97.4 ± 0.1 |
| 0.003 | 95.4 ± 0.2 | 97.7 ± 0.2 | 98.3 ± 0.1 |
| 0.01 | 96.5 ± 0.1 | 98.6 ± 0.0 | 98.8 ± 0.0 |
| 0.03 | **97.1 ± 0.1** | **98.7 ± 0.0** | **98.9 ± 0.2** |
| 0.1 | 96.1 ± 0.2 | 98.1 ± 0.1 | 98.7 ± 0.1 |
| 0.3 | 9.7 ± 0.5 | 9.7 ± 0.5 | 97.7 ± 0.3 |

*Final validation accuracy (%), mean ± standard deviation over three seeds.*

Three things stand out.

**The peak does not move.** Every model does best at $\eta = 0.03$. On this grid, half a decade
apart, the optimum is the same for an MLP with half a million parameters and a ResNet with eleven
million. Prediction 2 expected ResNet-18's peak further right. It wasn't.

**The width of the good region differs, and mostly on the low side.** Within one point of each
model's peak, the CNN and ResNet-18 both accept everything from 0.003 to 0.1, a 33× range. The
MLP accepts only 0.01 to 0.03. Its problem is the low end: at 0.001 it is five points short after
ten epochs, simply because it hasn't finished learning.

**The cliff moves.** At $\eta = 0.3$ the MLP and the CNN end at 9.7%, which is chance for ten
classes. ResNet-18 ends at 97.7%. Prediction 2 got this half right: BatchNorm did not move the
best rate, but it moved the point where training breaks.

### What dying looks like

{numref}`fig-lr-curves` shows the training loss on the fixed 5,000-image probe.

```{figure} figures/train-loss-curves.png
:alt: Training loss per epoch on a log scale, three panels for MLP, CNN and ResNet-18, one line per learning rate. In the MLP and CNN panels the 0.3 line is flat at about 2.3 from the first epoch. In the ResNet-18 panel every line decreases, with the 0.3 line highest.
:name: fig-lr-curves

Training loss per epoch (log scale), measured in eval mode on a fixed 5,000-image slice of the
training set. Darker lines are larger learning rates. In the MLP and CNN panels the $\eta = 0.3$
line is flat at $\ln 10$ from the first epoch: those networks output the same prediction for
every image. In the ResNet-18 panel even $\eta = 0.3$ keeps descending.
```

Back to the clue from the opening. For the MLP and the CNN at $\eta = 0.3$, the loss sits at
2.308 from epoch 1 to epoch 10. That number is not arbitrary: $\ln 10 = 2.303$ is the cross-entropy of predicting all ten classes with
equal probability. These networks didn't produce NaNs or oscillate. Within the first epoch a few
oversized steps threw them into a state where every image gets the same output, and they never
left it. The weight norm logged with every row shows how complete that is for the CNN. Its norm
jumps from about 9 to 649.5 during the first epoch and then does not change for nine epochs, so
nothing below the output layer is receiving gradient, which is what every ReLU switching off
would look like. The MLP's norm keeps drifting,
from 157 to 225, but the loss never moves off $\ln 10$, so whatever is still updating has no effect
on the output.

The ResNet-18 panel holds a quieter surprise. Its *lowest* training losses come from the
*smallest* learning rates: $1.5 \times 10^{-4}$ at 0.003, against $3.1 \times 10^{-3}$ at 0.03.
Its best validation accuracy comes from 0.03. Fitting the training set harder did not translate
into generalising better. Larger learning rates acting as a regulariser is a known effect
[^li19], and here it is worth about half a point of validation accuracy.

## Why ResNet-18 survives a learning rate that kills the others

The weight norms explain it. Here is the norm of all parameters, $\lVert\theta\rVert$, after the
first and the last epoch, averaged over seeds:

| Learning rate | 0.01 | 0.03 | 0.1 | 0.3 |
|---|---:|---:|---:|---:|
| MLP | 17.3 → 20.4 | 18.7 → 23.5 | 25.0 → 46.5 | 156.6 → 224.8 |
| CNN | 10.7 → 13.5 | 12.8 → 16.6 | 21.9 → 33.3 | 649.5 → 649.5 |
| ResNet-18 | 112.1 → 112.3 | 112.9 → 113.6 | 121.4 → 126.4 | 286.8 → 302.3 |

*$\lVert\theta\rVert$ after epoch 1 → after epoch 10. ResNet-18 starts at about 111.8.*

Every network's weights blow up at $\eta = 0.3$. For the MLP and the CNN, bigger weights mean a
bigger function, with larger pre-activations and larger logits, and the network falls off the edge.
For ResNet-18 the same blow-up is harmless, and in fact protective.

The reason is **scale invariance**. A convolution followed by BatchNorm computes the same output if
its weights $w$ are multiplied by any $c > 0$, because BatchNorm divides the scale right back out.
So the loss satisfies $\mathcal{L}(c\,w) = \mathcal{L}(w)$, and two consequences follow from
differentiating that identity:

$$
w \cdot \nabla_w \mathcal{L}(w) = 0,
\qquad
\nabla_w \mathcal{L}(c\,w) = \tfrac{1}{c}\,\nabla_w \mathcal{L}(w).
$$

The first says the gradient is always orthogonal to $w$. The second says it shrinks as $w$ grows.
{numref}`fig-lr-scale` draws what one plain SGD step does under those two facts.

```{figure} ../diagrams/scale-invariance.svg
:name: fig-lr-scale
:alt: Vector diagram: a weight vector w, a gradient step orthogonal to it, and the resulting longer vector w-prime, with the small angle theta between them. Side panel lists the scale-invariance identities and the resulting norm growth and angular step.

One SGD step on weights feeding a BatchNorm layer. Because the step is orthogonal to $w$, it can
only lengthen the vector. And because the function depends only on the direction $\hat w$, what
matters is the angle $\theta$, which shrinks with $\lVert w\rVert^2$.
```

Since the step is orthogonal to $w$, Pythagoras gives
$\lVert w'\rVert^2 = \lVert w\rVert^2 + \eta^2 \lVert\nabla\mathcal{L}(w)\rVert^2$, so every
step makes $w$ longer. The function only depends on the direction $\hat w = w / \lVert w\rVert$,
so the step that matters is the angle it turns through:

$$
\theta \;\approx\; \frac{\eta\,\lVert \nabla\mathcal{L}(w)\rVert}{\lVert w\rVert}
\;=\; \frac{\eta\,\lVert \nabla\mathcal{L}(\hat w)\rVert}{\lVert w\rVert^{2}}.
$$

The *effective* learning rate is $\eta / \lVert w\rVert^2$ [^laarhoven][^arora]. A learning rate
that is too large inflates $\lVert w\rVert$ in the first few hundred steps, and that inflation
divides the effective rate back down. The network corrects its own step size.

The logged norms let us size this, roughly. ResNet-18's norm goes from 111.8 to 286.8 during the
first epoch at $\eta = 0.3$. That divides the effective rate by $(286.8 / 111.8)^2 \approx 6.6$,
bringing 0.3 down to the neighbourhood of 0.05, close to the region where the model trains best.
Treat that as a back-of-envelope figure, not a measurement. $\lVert\theta\rVert$ here covers
*all* parameters, including the final linear layer and BatchNorm's own scale and shift, which are
not scale-invariant, and momentum blurs the clean geometry. But the direction and the size of the
effect match what {numref}`fig-lr-accuracy` shows. The cost is visible too: at 0.3, ResNet-18's
first-epoch validation accuracy is 78%, against 97% at 0.03. It loses that first epoch to
inflating its own weights, and spends the remaining nine catching up to within a point.

## Reading the stability bound backwards

The MLP and the CNN train well at 0.1 and are dead at 0.3. If the quadratic bound
$\eta\lambda_{\max} < 3.8$ were the whole story, their early-training curvature would sit between
$3.8/0.3 \approx 13$ and $3.8/0.1 = 38$. I would not lean on those numbers. Curvature during
training is not fixed: gradient descent tends to push the sharpness up until it sits right at the
stability edge [^cohen]. Even so, the training curves are consistent with this picture. At
$\eta = 0.1$ the MLP's and the CNN's losses stall around $10^{-2}$ and wobble. Those runs are
already running close to the edge.

## Validation versus test

The best rate per model, picked on validation, then scored once on the test set:

| Model | Best rate | Train acc. | Validation acc. | Test acc. |
|---|---:|---:|---:|---:|
| MLP | 0.03 | 100.0 | 97.1 | 91.7 |
| CNN | 0.03 | 100.0 | 98.7 | 95.8 |
| ResNet-18 | 0.03 | 99.9 | 98.9 | 96.4 |

*Accuracy (%) after 10 epochs, mean of three seeds. Training accuracy is on the 5,000-image probe.*

Test accuracy is 2.5 to 5.5 points below validation for every model, and the gap is not a
learning-rate artifact. It shows up at every rate, and it is widest at 0.001, where the MLP drops
10.3 points. The validation images are drawn from the same 60,000-image pool as the training
images, while the test set is Kuzushiji-MNIST's own separate set of 10,000, so the gap measures
how much those two sources differ, not how much the models overfit. Validation still did its job.
The ranking of the models, and of the learning rates within each model, is the same on both sets,
apart from one tie (the CNN scores 95.0% on test at both 0.01 and 0.1). But the absolute
validation numbers are optimistic, and a reader comparing them with published test-set results
should use the test column.

## Checking my guesses

| Prediction | Verdict |
|---|---|
| 1. Every model has an interior optimum | **Confirmed.** All three peak at 0.03, inside the grid. |
| 2. ResNet-18's curve sits further right and is wider | **Half right.** Same peak. Its tolerance extends further right (97.7% at 0.3, where the others are at chance), but the CNN's good region is as wide below the cliff. |
| 3. Ten epochs at 0.001 are not enough for any model | **Wrong for ResNet-18.** It reaches 100% training accuracy at 0.001. The MLP and the CNN are still descending. |

## The short version

On this dataset, with momentum SGD and a constant rate, the best learning rate was a property of
the optimiser and the problem rather than of the architecture: 0.03 for an MLP, a small CNN and
ResNet-18 alike. What the architecture decided was the *failure mode*. Without normalisation, a
learning rate one step too large kills a network in its first epoch, silently, with a loss of
exactly $\ln 10$. With BatchNorm, the same mistake inflates the weights, the effective step shrinks
in proportion, and the network recovers to within a point.

The practical lesson is about which side of the peak to err on. For un-normalised networks the
peak sits close to a cliff, so err low. For BatchNorm networks the cliff is far away, and the
small-rate side is where they underperform, fitting the training set more tightly and validating
worse.

Two limits on all of this. Ten epochs at a constant rate is a short, schedule-free budget. Warmup
or decay would soften the cliff and probably lift the small rates. And Kuzushiji-MNIST is one
dataset. The scale-invariance mechanism is general. The fact that all three peaks landed on the
same grid point may not be.

[^bengio]: Y. Bengio, "Practical Recommendations for Gradient-Based Training of Deep
    Architectures," in *Neural Networks: Tricks of the Trade*, 2nd ed., Springer, 2012.
[^goh]: G. Goh, "Why Momentum Really Works," *Distill*, 2017.
[^bjorck]: N. Bjorck, C. P. Gomes, B. Selman, K. Q. Weinberger, "Understanding Batch
    Normalization," NeurIPS 2018.
[^santurkar]: S. Santurkar, D. Tsipras, A. Ilyas, A. Mądry, "How Does Batch Normalization Help
    Optimization?," NeurIPS 2018.
[^li19]: Y. Li, C. Wei, T. Ma, "Towards Explaining the Regularization Effect of Initial Large
    Learning Rate in Training Neural Networks," NeurIPS 2019.
[^laarhoven]: T. van Laarhoven, "L2 Regularization versus Batch and Weight Normalization,"
    arXiv:1706.05350, 2017.
[^arora]: S. Arora, Z. Li, K. Lyu, "Theoretical Analysis of Auto Rate-Tuning by Batch
    Normalization," ICLR 2019.
[^cohen]: J. M. Cohen, S. Kaur, Y. Li, J. Z. Kolter, A. Talwalkar, "Gradient Descent on Neural
    Networks Typically Occurs at the Edge of Stability," ICLR 2021.
[^kmnist]: T. Clanuwat, M. Bober-Irizar, A. Kitamoto, A. Lamb, K. Yamamoto, D. Ha, "Deep Learning
    for Classical Japanese Literature," NeurIPS 2018 Workshop on Machine Learning for Creativity
    and Design.
