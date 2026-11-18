---
title: Batch Size Versus Training Time
subtitle: Which batch size wins depends on the clock you hold, whether that's epochs, updates or seconds. EuroSAT satellite patches.
date: 2026-11-18
authors:
  - name: Eleijahmar Fifth C. Sevare
tags: [deep learning, batch size, pytorch, eurosat]
---

Give ResNet-18 a batch size of 16 and it wins: 89.0% on EuroSAT after ten epochs, the best score
in this post. It also needs 131 seconds of training. Give it a batch of 256 and it reaches 80% in
about 10 seconds, then finishes eight points lower.

So which batch size is better? It depends on which clock you are holding, and this post holds
three: **epochs**, **updates** and **seconds**. Batch size is usually picked to fill the GPU, and
it looks like an engineering setting. Changing it turns three dials at once: how noisy each
gradient step is, how many steps an epoch contains, and, in a network with BatchNorm, how the
model normalises its own activations.

> **Question.** With the learning rate held fixed, how does batch size change accuracy and
> training time for an MLP, a small CNN and ResNet-18? Does the network with BatchNorm react
> differently from the two without it?

On the epoch clock the three models disagreed completely. The MLP could not learn with the
smallest batch, the CNN peaked at 64, and ResNet-18 peaked at 16, with larger-batch runs that turned
out to be unstable in a way the headline plot hides. On the update clock they agreed: at an equal
number of updates, the bigger batch was ahead in every comparison. On the wall clock, a middle
batch size got to the target first for every model.

## The data: Europe from 786 km up

EuroSAT [^eurosat] is 27,000 patches of Sentinel-2 satellite imagery from 34 European countries,
each labelled with one of ten land-use classes. Sentinel-2 orbits at 786 km and records 13
spectral bands. The torchvision version keeps the three visible ones. One pixel is 10 m of ground,
so a 64 × 64 patch covers 640 m × 640 m.

```{figure} figures/dataset-eurosat.png
:name: fig-bs-data
:alt: A grid of EuroSAT patches, ten per class, in rows labelled AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River and SeaLake with each class's image count. Forest and SeaLake rows are nearly uniform dark green and dark blue; Highway and River rows show thin lines across fields.

Ten random training patches per class, with the class size in brackets. Drawn from the raw
dataset files by `diagrams/datasets.py`.
```

Three things in {numref}`fig-bs-data` matter later. Colour alone nearly settles some classes:
Forest and SeaLake are close to uniform patches. Other classes are defined by thin structures, such
as a highway or a river a few pixels wide, which only exist as *shapes*. Flatten a patch into
12,288 numbers for the MLP and the shapes are gone, which is part of why the MLP finishes 20 points
behind the convolutional models. And at 10 m per pixel, Pasture, HerbaceousVegetation and
PermanentCrop are all green-brown fields. The classes are mildly imbalanced, with 2,000 to 3,000
patches each.

## Three things a batch size changes

Each SGD step estimates the full gradient from a mini-batch of $B$ examples:

$$
\hat{g}_B = \frac{1}{B}\sum_{i \in \mathcal{B}} \nabla_\theta\, \ell(x_i, y_i; \theta),
\qquad
\operatorname{Var}\big[\hat{g}_B\big] \approx \frac{\Sigma}{B},
$$

where $\Sigma$ is the covariance of single-example gradients. A bigger batch gives a cleaner
estimate. But an epoch over $N$ images has only $N/B$ steps, so with the learning rate $\eta$
fixed, going from $B = 16$ to $B = 1024$ cuts the number of updates per epoch by 64×.

Smith and Le [^smithle] fold both effects into a single *noise scale* for SGD with momentum $\mu$:

$$
g \approx \frac{\eta}{1-\mu}\cdot\frac{N}{B}.
$$

With $\eta = 0.01$, $\mu = 0.9$ and $N = 21{,}600$, the four batch sizes here span a 64× range of
noise: $g \approx 135$, $34$, $8.4$ and $2.1$. Large-batch training tends to settle in sharper
minima that generalise worse [^keskar]. Much of that gap closes if large batches get the same
number of *updates* rather than the same number of epochs [^hoffer], which is why the
[linear scaling rule](https://arxiv.org/abs/1706.02677) [^goyal] multiplies $\eta$ along with $B$.
This experiment deliberately does **not** do that. The learning rate stays at 0.01, so the sweep
measures what happens when you change the batch size and nothing else, as most people do.

The hardware sees batch size differently again. A GPU processes a batch of 64 in little more time
than a batch of 16, so larger batches finish an epoch sooner. The fair comparison is accuracy
against seconds, not only against epochs.

## Where the models differ: BatchNorm makes the batch part of the model

{numref}`fig-bs-models` shows the three architectures. For the MLP and the CNN, $B$ changes only
the gradient noise and the update count. ResNet-18 is different.

```{figure} ../diagrams/models.svg
:name: fig-bs-models
:alt: Three layer stacks side by side: MLP, CNN and ResNet-18. ResNet-18's stem and four residual stages are shaded to mark BatchNorm; the MLP and CNN have no shaded boxes.

The three models. Shaded boxes contain BatchNorm, and only ResNet-18 has any. Dropout layers
(dashed) are identity functions in this post. Parameter counts are for the 10 EuroSAT classes,
listed under each model.
```

ResNet-18 normalises every convolution's output with the mean and variance of the *current
batch* [^ioffe]. {numref}`fig-bs-resnet` shows where those statistics are computed: per channel,
over the $B \times H \times W$ positions in the batch.

```{figure} ../diagrams/resnet18.svg
:name: fig-bs-resnet
:alt: ResNet-18 stage diagram with spatial sizes per dataset (EuroSAT 64 down to 2 after layer4); below, a BasicBlock with two BatchNorm layers and a note that BatchNorm uses the current mini-batch statistics in training and running averages in evaluation.

ResNet-18 as used here. On 64 × 64 EuroSAT patches, the map is still 2 × 2 after `layer4`. Each
BasicBlock contains two BatchNorm layers. In training they normalise with the current batch's
statistics, so $B$ is also the sample size of every normalisation. In evaluation they use
running averages accumulated during training.
```

So for ResNet-18, $B$ is also the sample size of 20 normalisation layers. Small batches make the
normalisation noisy, which can act as a regulariser at moderate sizes and becomes a handicap when
batches get very small [^wu]. There is a second, subtler consequence. Training uses per-batch
statistics, while evaluation uses running averages collected during training. When those two
disagree, a model can behave differently at evaluation time from how it behaved while training.
That turns out to matter below.

## Timing it honestly

The harness is shared by every post in the series ({numref}`fig-bs-harness`): one fixed split,
data resident on the GPU, and the same initial weights for every batch size within a seed.

```{figure} ../diagrams/harness.svg
:name: fig-bs-harness
:alt: Flow diagram of the experiment: data decoded, split and normalised once and kept on the GPU; a sweep over value, model and seed; each run seeds the model, then per epoch shuffles, trains with SGD inside a timed region, scores a train probe and validation in eval mode, and logs a row; test accuracy is scored once at the end; outputs are runs.csv, figures and results.md.

The experiment harness. The timer wraps only the SGD steps. Evaluation passes are excluded, and
because batches are index slices of tensors already on the GPU, no data loading is timed either.
```

Timing needs one detail done right. CUDA kernels run asynchronously, so a timer read without
synchronising would stop before the GPU finished. `run()` synchronises before reading the clock:

```python
model.train()
t0 = time.time()
order = torch.randperm(len(x), device=DEVICE, generator=order_rng)
for i in range(0, len(x) - batch_size + 1, batch_size):   # drops the last partial batch
    idx = order[i:i + batch_size]
    loss = F.cross_entropy(model(x[idx]), y[idx])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
if DEVICE == "cuda":
    torch.cuda.synchronize()
train_time += time.time() - t0
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Reserted/Data-Science-Portfolio/blob/main/blog/batch-size/batch-size-colab.ipynb)

Every figure and number below comes from [`batch-size-colab.ipynb`](batch-size-colab.ipynb),
written for a free Colab T4 GPU. Absolute seconds depend on the GPU. The ratios between batch
sizes are the part worth reading.

| Setting | Value |
|---|---|
| Dataset | EuroSAT [^eurosat] (`torchvision.datasets.EuroSAT`): 27,000 Sentinel-2 RGB patches, 64×64, 10 land-use classes |
| Split | no official split: fixed random 21,600 train / 2,700 validation / 2,700 test |
| MLP | 12,288 → 512 → 256 → 10, ReLU; 6,425,866 parameters |
| CNN | two 3×3 conv blocks (32, 64 channels) with max-pool → dense 128 → 10; 2,117,962 parameters |
| ResNet-18 | `torchvision.models.resnet18`, random init; 11,181,642 parameters |
| Optimiser | SGD, momentum 0.9, learning rate 0.01 for every batch size |
| Held fixed | no dropout, no weight decay, no augmentation |
| **Swept** | **batch size ∈ {16, 64, 256, 1024}** → 1,350 / 337 / 84 / 21 steps per epoch |
| Runs | 4 batch sizes × 3 models × 3 seeds = 36 runs, 10 epochs each |
| Measured | final-epoch validation accuracy, training seconds per epoch; test accuracy for each model's best batch size |

```python
df = sweep("batch_size", [16, 64, 256, 1024], epochs=10)
```

## Three bets

Placed before any run:

1. Per epoch, accuracy falls as the batch grows. At $B = 1024$ the whole run is only 210
   updates, which should leave every model under-trained.
2. Time per epoch falls as the batch grows, most of all for ResNet-18, which wastes the most GPU
   time on tiny batches.
3. Against wall-clock time, a middle batch size (64 or 256) gets to a given accuracy first.

## Clock 1: epochs

This is the clock most training scripts hold: run ten epochs, report the last one.
{numref}`fig-bs-accuracy` shows that result.

```{figure} figures/accuracy-vs-batch-size.png
:alt: Final validation accuracy against batch size, one line per model, with seed bands. MLP rises from 0.38 at 16 to 0.65 at 256. CNN peaks at 0.85 at 64. ResNet-18 is highest at 16 (0.89), dips to 0.74 at 64 with a wide band, and recovers to about 0.81.
:name: fig-bs-accuracy

Final validation accuracy after 10 epochs, learning rate fixed at 0.01. Lines are the mean of
three seeds, and bands span the worst and best seed. The three models disagree completely about
which batch size is best. ResNet-18's wide bands at $B \geq 64$ are the first sign that its
final-epoch numbers are unreliable.
```

| Batch size | MLP | CNN | ResNet-18 |
|---:|:---:|:---:|:---:|
| 16 | 37.9 ± 2.9 | 77.3 ± 2.8 | **89.0 ± 0.1** |
| 64 | 62.0 ± 1.3 | **85.3 ± 0.6** | 73.6 ± 2.9 |
| 256 | **65.0 ± 0.7** | 84.4 ± 0.7 | 81.2 ± 4.7 |
| 1024 | 62.2 ± 0.5 | 75.9 ± 0.4 | 81.0 ± 4.4 |

*Final validation accuracy (%), mean ± standard deviation over three seeds.*

There is no common shape. The MLP gets *better* with batch size up to 256. The CNN has an inverted
U with its top at 64. ResNet-18 does best at the smallest batch, with a seed spread of 0.1 points,
then drops 15 points and becomes erratic. Before explaining any of this, it is worth checking
whether these final-epoch numbers mean what they appear to.

### The final epoch is not the whole story

{numref}`fig-bs-epoch` shows the same runs epoch by epoch.

```{figure} figures/accuracy-per-epoch.png
:alt: Validation accuracy per epoch, three panels (MLP, CNN, ResNet-18), one line per batch size. The MLP at batch size 16 stays between 0.34 and 0.40 throughout. In the ResNet-18 panel the batch-16 line rises steadily to 0.89 while the batch-64 line peaks near 0.85 at epoch 5-7 and falls to 0.74 at epoch 10.
:name: fig-bs-epoch

Validation accuracy per epoch, one panel per model, mean of three seeds. Darker lines are larger
batches. Look at ResNet-18's $B = 64$ line: it climbs to about 85% by epoch 5, then loses more than
ten points over the last three epochs. The $B = 16$ line rises almost monotonically.
```

ResNet-18's dip at $B = 64$ is not a property of that batch size. It is what the last epoch
happened to catch. Looking at the three seeds separately, every one of them was above 75% at
epoch 9 and fell at epoch 10:

| ResNet-18, B = 64 | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| best epoch | 87.8 (ep 6) | 87.2 (ep 7) | 86.5 (ep 7) |
| epoch 9 | 87.2 | 81.8 | 75.9 |
| epoch 10 | 76.3 | 74.0 | 70.5 |
| validation loss at epoch 10 | 1.23 | 1.07 | 2.68 |

*Validation accuracy (%) per seed.*

This is not overfitting. The eval-mode accuracy on the 5,000-image training probe falls in the
same epochs, ending at 87.2%, 84.8% and 79.6%. The network got worse at its *own training data*.
Two explanations fit the logs, and I can't separate them with what was recorded. Either a constant
learning rate with momentum keeps throwing these runs back out of good regions, or the running
BatchNorm statistics used at evaluation lag behind weights that are still moving fast. What the logs
do show is that $B = 16$ is the only setting where ResNet-18 is stable: all three seeds end within
0.2 points of each other.

Scoring each run by its *best* epoch instead of its last gives a cleaner picture:

| Batch size | MLP | CNN | ResNet-18 |
|---:|:---:|:---:|:---:|
| 16 | 42.4 | 80.3 | 89.5 |
| 64 | 64.2 | 85.6 | 87.2 |
| 256 | 65.4 | 84.8 | 85.3 |
| 1024 | 62.3 | 76.4 | 84.0 |

*Best-epoch validation accuracy (%), mean over seeds. This peeks at validation every epoch, so it
is a diagnostic, not a fair final score.*

ResNet-18 now declines smoothly with batch size, which is the textbook shape. The MLP and the CNN
barely change, because their curves were not collapsing at the end.

### The MLP at B = 16 isn't overfitting. It can't fit.

The MLP's 37.9% at $B = 16$ looks like the other extreme, a model that learned nothing useful.
The training probe shows that is literally true. Its eval-mode training accuracy is 41.8%,
barely above its validation accuracy. This is an optimisation failure, not a generalisation one.
With a noise scale of $g \approx 135$ and no normalisation anywhere, each step on a
12,288-dimensional input is mostly noise. In {numref}`fig-bs-epoch` its validation accuracy
wanders between 34% and 40% for ten epochs without trending upward. The CNN suffers the same
problem in milder form: it reaches the 80% mark at $B = 16$ in only two of three seeds.

## Clock 2: updates

The cleanest result in the experiment comes from a clock the figures don't show directly.
Because the number of updates per epoch is known exactly, runs with different batch sizes can be
compared at the *same number of updates*:

| Updates | Smaller batch | Larger batch | MLP | CNN | ResNet-18 |
|---:|---|---|:---:|:---:|:---:|
| ≈ 1,350 | B = 16, epoch 1 | B = 64, epoch 4 | 35.0 → 62.7 | 65.7 → 83.5 | 71.6 → 81.2 |
| ≈ 337 | B = 64, epoch 1 | B = 256, epoch 4 | 56.5 → 63.0 | 72.4 → 77.9 | 71.9 → 80.9 |
| 168 | B = 256, epoch 2 | B = 1024, epoch 8 | 58.4 → 60.4 | 71.6 → 75.3 | 68.7 → 80.3 |

*Validation accuracy (%), smaller batch → larger batch, at the same update count. Mean of three
seeds.*

In all nine comparisons the larger batch is ahead. That is exactly what the variance formula
predicts: same number of steps, cleaner gradient per step. So the per-epoch losses of the large
batches in {numref}`fig-bs-accuracy` are a budget problem rather than a quality problem. At
$B = 1024$ the entire run is 210 updates, and the models simply stop before they are done.
ResNet-18 at $B = 16$ wins the per-epoch comparison the opposite way. It is *less* efficient per
step (71.6% after 1,350 updates, against 81.2% for $B = 64$), but it gets 13,500 of them and stays
stable for all ten epochs.

## Clock 3: seconds

This is the clock you pay for. {numref}`fig-bs-time` replots every run against training time.

```{figure} figures/accuracy-per-second.png
:alt: Validation accuracy against cumulative training seconds, three panels, one line per batch size. MLP runs finish within about 4 seconds except batch 16 at about 17 seconds. CNN batch 64 reaches 0.85 in about 10 seconds; batch 1024 is slowest to rise. ResNet-18 batch 16 runs for about 130 seconds, the others finish in 25 to 35 seconds.
:name: fig-bs-time

The same runs against training time instead of epochs. Each line ends when its ten epochs end, so
a shorter line means a faster run. ResNet-18 at $B = 16$ reaches the highest accuracy, but it takes
about five times as long to get there.
```

| Batch size | MLP | CNN | ResNet-18 |
|---:|---:|---:|---:|
| 16 | 1.66 | 2.45 | 13.08 |
| 64 | 0.42 | **1.06** | 3.42 |
| 256 | 0.13 | 1.54 | **2.56** |
| 1024 | **0.09** | 1.58 | 2.65 |

*Training seconds per epoch, mean over seeds. Evaluation excluded.*

Larger batches make the MLP 19× faster per epoch and ResNet-18 5× faster, with ResNet-18 flat from
256 onward because it has saturated the GPU. The CNN is the odd one. It is fastest at 64 and gets
*slower* at 256 and 1024. I did not profile this. A plausible cause is that the convolution
algorithm PyTorch picks by default (benchmark mode is off) suits some batch shapes much better than
others, but that is a hypothesis, not a finding.

Combining speed and accuracy, here is the time to first reach a fixed validation accuracy (median
of three seeds):

| Model, target | B = 16 | B = 64 | B = 256 | B = 1024 |
|---|---:|---:|---:|---:|
| MLP, 60% | never | 1.2 s | **0.4 s** | 0.7 s |
| CNN, 80% | 2 of 3 seeds | **3.2 s** | 9.2 s | never |
| ResNet-18, 80% | 26.3 s | 13.7 s | **10.3 s** | 15.9 s |

A middle batch size gets there first for every model, as predicted. For ResNet-18 that holds only
for the 80% target. Its highest accuracy still belongs to $B = 16$, if you pay for it.

## Validation and test agree here

For the best batch size per model, test accuracy matches validation to within one point:

| Model | Best B | Train acc. | Validation acc. | Test acc. |
|---|---:|---:|---:|---:|
| MLP | 256 | 79.1 | 65.0 | 64.1 |
| CNN | 64 | 98.7 | 85.3 | 85.6 |
| ResNet-18 | 16 | 97.7 | 89.0 | 88.4 |

*Accuracy (%), mean of three seeds; training accuracy on the 5,000-image probe.*

That is a property of the split, not the models. EuroSAT has no official test set, so validation
and test are both random draws from the same pool, and they behave alike. (In the
[learning-rate post](../learning-rate/learning-rate.md), where Kuzushiji-MNIST's separate test set was used, test
ran 2.5 to 5.5 points below validation.) The MLP's 14-point gap between training and validation at
its best setting is the one real generalisation gap here. Flattening a satellite patch into 12,288
numbers throws away the spatial structure that the convolutional models use.

## Settling the bets

| Prediction | Verdict |
|---|---|
| 1. Per epoch, accuracy falls as the batch grows; B = 1024 under-trained | **Mixed.** True for the CNN above 64, and for ResNet-18 once its final-epoch collapse is set aside. False for the MLP, which is worst at 16. B = 1024 is under-trained for the CNN (75.9% vs 85.3%). |
| 2. Time per epoch falls with batch size, most for ResNet-18 | **Half right.** The MLP gains most (19×), ResNet-18 5×. The CNN is fastest at 64. |
| 3. A middle batch size reaches a given accuracy first | **Confirmed** for all three models (MLP 256, CNN 64, ResNet-18 256). |

## Which clock to hold

At a fixed learning rate, batch size trades three things: update count, gradient noise and, for
BatchNorm networks, normalisation noise. The three models weighed those trades differently. The
MLP needed clean gradients more than anything, and small batches starved it. The CNN liked the
balance at 64. ResNet-18 did best with the smallest batch and the most updates, and was the only
model whose larger-batch runs were unstable late in training. The one regularity that held
everywhere was that per update, the larger batch was ahead. When a large batch "loses", it is
almost always because it ran out of steps.

The practical reading is that batch size and learning rate should not be tuned separately. Holding
$\eta$ fixed, as this experiment does on purpose, gave $B = 1024$ only 210 steps of the same size.
Scaling $\eta$ with $B$ [^goyal] and training until convergence rather than for a fixed ten epochs
would likely flatten most of these curves, and I'd expect the gap between batch sizes to shrink to
the wall-clock column. Two limits on everything above: one dataset, and ten epochs at a constant
learning rate. Both the late instability of ResNet-18 and the CNN's timing anomaly deserve a run
with a learning-rate schedule and a profiler before anyone generalises from them.

[^smithle]: S. L. Smith, Q. V. Le, "A Bayesian Perspective on Generalization and Stochastic
    Gradient Descent," ICLR 2018.
[^keskar]: N. S. Keskar, D. Mudigere, J. Nocedal, M. Smelyanskiy, P. T. P. Tang, "On Large-Batch
    Training for Deep Learning: Generalization Gap and Sharp Minima," ICLR 2017.
[^hoffer]: E. Hoffer, I. Hubara, D. Soudry, "Train Longer, Generalize Better: Closing the
    Generalization Gap in Large Batch Training of Neural Networks," NeurIPS 2017.
[^goyal]: P. Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour,"
    arXiv:1706.02677, 2017.
[^ioffe]: S. Ioffe, C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift," ICML 2015.
[^wu]: Y. Wu, K. He, "Group Normalization," ECCV 2018.
[^eurosat]: P. Helber, B. Bischke, A. Dengel, D. Borth, "EuroSAT: A Novel Dataset and Deep
    Learning Benchmark for Land Use and Land Cover Classification," *IEEE Journal of Selected
    Topics in Applied Earth Observations and Remote Sensing*, 12(7), 2019.
