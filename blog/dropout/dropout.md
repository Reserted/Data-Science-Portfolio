---
title: Dropout Stablizes Confidence, Not Accuracy
subtitle: I picked USPS to hand dropout an easy win. How accuracy is not the end all be all
date: 2026-11-18
authors:
  - name: Eleijahmar Fifth C. Sevare
tags: [deep learning, dropout, regularization, pytorch, usps]
---

I chose this dataset to give dropout the easiest win I could find. USPS has 7,291 training
images, few enough for any of my three models to memorise, and memorisation is what dropout
exists to stop. The models did memorise it, and dropout barely made a difference. Across every
rate and every model, validation accuracy moved by a point at most.

That would make a short post, except that the accuracy plot turned out to be the least
interesting of three. On the CNN, dropout cut the validation *loss* by more than a third and left
validation *accuracy* exactly where it was. It didn't change what the network decided. It changed
how sure the network was when it was wrong.

> **Question.** On a small dataset that every model can memorise, how does the dropout rate
> change accuracy and overfitting for an MLP, a small CNN and ResNet-18?

Part of the answer is where the dropout layers sit. Introductory code adds dropout almost by
reflex, usually with `p=0.5`, but dropout is a layer rather than a setting, and its position
decides what it can reach. The same $p$ means something different in a plain MLP than in a ResNet
that already has BatchNorm everywhere.

## The data: ZIP codes from the US mail

USPS [^usps] is an old dataset. Its 9,298 digits were cut from handwritten ZIP codes on envelopes
in the US mail, and they are the digits LeCun and colleagues trained their 1989 zip-code network
on [^lecun], one of the first convolutional networks trained with backpropagation.

```{figure} figures/dataset-usps.png
:name: fig-do-data
:alt: A grid of USPS training digits, fourteen per class, in rows 0 to 9 with each digit's image count: 0 has 1,194, 1 has 1,005, down to 8 with 542. The digits are thick, blurry and fill their frames.

Fourteen random training images per digit, with the class size in brackets. Drawn from the raw
dataset files by `diagrams/datasets.py`.
```

{numref}`fig-do-data` shows what makes it easy. Every digit has been scaled to fill its
16 × 16 frame, so there is no empty margin and little variation in size or position. The strokes
are thick and the handwriting is mostly tidy. The classes are not balanced, though. There are
1,194 zeros and only 542 eights.

```{figure} figures/dataset-usps-pixels.png
:name: fig-do-pixels
:alt: One USPS training image of the digit 5, magnified to a 16 by 16 grid with every pixel's value printed, from −1 for background to about 0.7 for ink.

One training 5, with every pixel's value printed. The file stores pixels from −1 (background)
to 1 (ink), and torchvision rescales them to 0–255. These 256 numbers, flattened into one row,
are the MLP's entire input.
```

An image this small can be read in full, number by number ({numref}`fig-do-pixels`). The small
size also matters for ResNet-18, which was designed for 224 × 224 photos. Its five stride-2 steps
reduce this grid to a single pixel well before the network ends. That comes back in the diagram
below.

## What dropout does

During training, dropout switches each unit off with probability $p$ and scales the survivors so
that the expected activation stays the same [^srivastava]:

$$
\tilde{h}_j = \frac{m_j\, h_j}{1-p}, \qquad m_j \sim \operatorname{Bernoulli}(1-p),
\qquad \mathbb{E}\big[\tilde{h}_j\big] = h_j.
$$

No unit can count on a particular partner being present, so the network can't memorise the
training set through fragile co-adaptations between units. Each step trains a different thinned
sub-network. At test time dropout is switched off, which approximates averaging that whole
ensemble [^srivastava]. The price is noisier, slower training, and with too much dropout the
network can no longer fit even its training data.

That last point creates a measurement trap. The training accuracy a progress bar shows is computed
*with dropout on*, so it understates how well the network fits its own data, and by more the higher
$p$ is. Comparing it with validation accuracy measured with dropout off would make every
high-dropout model look like it generalises beautifully. This experiment avoids that. Every epoch
it re-scores a fixed slice of the training set in evaluation mode:

```python
@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    """Mean cross-entropy and accuracy in eval mode (dropout off, BatchNorm running statistics)."""
    model.eval()
    loss = correct = 0.0
    for i in range(0, len(x), bs):
        out = model(x[i:i + bs])
        loss += F.cross_entropy(out, y[i:i + bs], reduction="sum").item()
        correct += (out.argmax(1) == y[i:i + bs]).sum().item()
    return loss / len(x), correct / len(x)

train_loss, train_acc = evaluate(model, x[:5000], y[:5000])   # fixed training slice, dropout off
```

Train and validation accuracy are then measured the same way, and their difference, the
*generalisation gap*, is a fair number.

## Where the dropout layers sit

Dropout only acts where a dropout layer is placed, and the three models place it very differently.
{numref}`fig-do-models` shows each stack. In this post the dashed boxes are live.

```{figure} ../diagrams/models.svg
:name: fig-do-models
:alt: Three layer stacks. MLP: dropout after each of its two hidden layers. CNN: dropout after the flatten and after the dense 128 layer, both in the dense head. ResNet-18: one dropout just before the final linear layer, after global average pooling.

The three models. Dashed boxes are the dropout layers, all set to the same $p$ in each run. The MLP
has dropout after both hidden layers, which covers almost every weight. The CNN has it only in
its dense head, so the convolutions are untouched. ResNet-18 has a single dropout in front of its
final 512 → 10 layer. Parameter counts for USPS are listed under each model.
```

| Model | Dropout layers | What they touch |
|---|---|---|
| MLP | after both hidden layers | almost every weight in the network |
| CNN | before both dense layers | the dense head; the convolutions are untouched |
| ResNet-18 | before the final classifier only | one 512 → 10 layer out of 11 million parameters |

ResNet-18 has no dropout of its own. Its regularisation comes from BatchNorm's batch noise. Mixing
the two can even hurt, because dropout shifts the activation variance that BatchNorm learned during
training [^li]. {numref}`fig-do-resnet` shows why the placement used here avoids that: the dropout
layer sits after the last BatchNorm and after global pooling, so no normalisation layer ever sees
its noise.

```{figure} ../diagrams/resnet18.svg
:name: fig-do-resnet
:alt: ResNet-18 stage diagram with spatial sizes; for USPS the map is 1x1 after layer3, and layer4 is marked red. Dropout sits after the average pool and before the linear layer. Below, a BasicBlock with two BatchNorm layers.

ResNet-18 as used here. Dropout comes after the last BatchNorm and the global average pool, so it
never disturbs a normalisation layer's statistics. On 16 × 16 USPS digits the feature map is
already 1 × 1 after `layer3`. The last stage, about three quarters of the network's
parameters, works on single pixels.
```

The diagram also exposes a quirk of using an ImageNet architecture on 16 × 16 images. Each of the
five stride-2 steps halves the resolution, so `layer4` receives a 1 × 1 map, and its 3 × 3 kernels
only ever use their centre weight. That doesn't break anything, but it means ResNet-18 here is
effectively a much shallower network with a lot of idle capacity, which is worth remembering when
it behaves differently from the others.

## The rig

The harness is shared by every post in the series ({numref}`fig-do-harness`).

```{figure} ../diagrams/harness.svg
:name: fig-do-harness
:alt: Experiment flow: data prepared once and kept on the GPU; a seeded sweep over value, model and seed; per epoch shuffle, SGD, eval-mode scoring of a train probe and the validation set; test scored once; outputs runs.csv, figures, results.md.

The experiment harness. Within a seed, every dropout rate starts from the same initial weights
and sees the same batch order, so differences between rates come from $p$ alone. The "train probe"
is the first 5,000 training images, scored in eval mode with dropout off.
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Reserted/Data-Science-Portfolio/blob/main/blog/dropout/dropout-colab.ipynb)

Every figure and number below comes from [`dropout-colab.ipynb`](dropout-colab.ipynb), written
for a free Colab T4 GPU.

| Setting | Value |
|---|---|
| Dataset | USPS [^usps] (`torchvision.datasets.USPS`): 16×16 grayscale digits scanned from US mail, 10 classes |
| Split | 7,291 training images → 6,291 train / 1,000 validation (fixed); official 2,007 test |
| MLP | 256 → 512 → 256 → 10, ReLU; 265,482 parameters |
| CNN | two 3×3 conv blocks (32, 64 channels) with max-pool → dense 128 → 10; 151,306 parameters |
| ResNet-18 | `torchvision.models.resnet18`, random init, first conv takes 1 channel; 11,175,370 parameters |
| Optimiser | SGD, momentum 0.9, learning rate 0.01 |
| Held fixed | batch size 128, no weight decay, no augmentation |
| **Swept** | **dropout rate p ∈ {0, 0.25, 0.5, 0.75}** |
| Runs | 4 rates × 3 models × 3 seeds = 36 runs, 30 epochs each |
| Measured | final-epoch validation accuracy and loss, train − validation gap; test accuracy for each model's best rate |

Thirty epochs is three times the budget of the other posts in the series, chosen so that
overfitting has time to show.

```python
df = sweep("dropout", [0.0, 0.25, 0.5, 0.75], epochs=30)
df["gap"] = df.train_acc - df.val_acc   # train accuracy is measured with dropout off
```

## Four predictions

Committed to before the first run:

1. The MLP gains the most from dropout: it has nothing else regularising it, and dropout touches
   all of it.
2. The train − validation gap shrinks as $p$ grows, for the MLP and the CNN.
3. ResNet-18 barely reacts. Dropout reaches only its last layer.
4. At $p = 0.75$ the MLP and CNN start to underfit: training accuracy drops along with
   validation accuracy.

## The accuracy plot: almost flat

{numref}`fig-do-accuracy` is the headline result. Before reading it, check the y-axis.

```{figure} figures/accuracy-vs-dropout.png
:alt: Final validation accuracy against dropout rate, one line per model, with seed bands. The y-axis spans 0.965 to 0.99. MLP rises from 0.971 to 0.976 at p=0.25 and falls to 0.966 at 0.75. CNN and ResNet-18 stay between 0.982 and 0.988 with overlapping bands.
:name: fig-do-accuracy

Final validation accuracy after 30 epochs against the dropout rate. Lines are the mean of three
seeds, and bands span the worst and best seed. The whole y-axis covers 2.5 points. On 1,000
validation images, 0.1 points is a single image.
```

| Dropout p | MLP | CNN | ResNet-18 |
|---:|:---:|:---:|:---:|
| 0 | 97.1 ± 0.1 | 98.4 ± 0.3 | 98.6 ± 0.5 |
| 0.25 | **97.6 ± 0.1** | **98.5 ± 0.2** | 98.7 ± 0.1 |
| 0.5 | 97.3 ± 0.2 | 98.4 ± 0.3 | **98.8 ± 0.1** |
| 0.75 | 96.6 ± 0.2 | 98.2 ± 0.3 | 98.4 ± 0.2 |

*Final validation accuracy (%), mean ± standard deviation over three seeds.*

The MLP has the only effect clearly larger than the seed noise. $p = 0.25$ adds half a point,
five more correct validation images, with a seed spread of 0.1. At $p = 0.75$ it ends half a
point *below* no dropout. The CNN and ResNet-18 move by 0.3 and 0.4 points across the entire range,
inside their own seed bands. Plotted on a 0–100% axis, all twelve configurations would look like
one flat line.

The reason is that there was little overfitting for dropout to remove. Even with no dropout, the
gap between training and validation accuracy is only 1.4 to 2.6 points:

```{figure} figures/gap-vs-dropout.png
:alt: Train minus validation accuracy against dropout rate. MLP falls from 0.026 at p=0 to about 0.020 at 0.25 and 0.5 and 0.018 at 0.75. CNN falls from 0.016 to 0.009. ResNet-18 stays near 0.013 and rises to 0.016 at 0.75.
:name: fig-do-gap

Generalisation gap: training accuracy (dropout off, 5,000-image probe) minus validation accuracy.
Lower means less overfitting. Dropout does narrow the gap for the MLP and the CNN, but the gap
was never large.
```

| Dropout p | MLP | CNN | ResNet-18 |
|---:|:---:|:---:|:---:|
| 0 | 2.6 | 1.6 | 1.4 |
| 0.25 | 2.0 | 1.3 | 1.3 |
| 0.5 | 2.0 | 1.3 | 1.2 |
| 0.75 | 1.8 | 0.9 | 1.6 |

*Train − validation accuracy gap (points), mean of three seeds.*

The gap does shrink as $p$ grows for the MLP and the CNN, as predicted. But at $p = 0.75$ it
shrinks for the wrong reason. The MLP's training accuracy falls from 99.8% to 98.4%, and its
validation accuracy falls too, from 97.1% to 96.6%. A smaller gap is only good news when
validation goes up. Here the network is simply fitting everything less well.

### Who actually memorised the training set

USPS is small, but it is easy, and the three models fit it at very different speeds:

| Model | p = 0 | p = 0.25 | p = 0.5 | p = 0.75 | Reaches 99.9% (p = 0) |
|---|---:|---:|---:|---:|---|
| MLP | 99.8 | 99.6 | 99.3 | 98.4 | never in 30 epochs |
| CNN | 100.0 | 99.9 | 99.7 | 99.1 | epochs 24–29 |
| ResNet-18 | 100.0 | 100.0 | 100.0 | 100.0 | epochs 6–8 |

*Final training accuracy (%) on the 5,000-image probe, dropout off. Last column: first epoch at
or above 99.9%, per seed.*

ResNet-18 memorises the training set at every dropout rate. Its single dropout layer in front of
the classifier delays that only modestly, from epochs 6–8 at $p = 0$ to epochs 11–14 at
$p = 0.75$. The CNN gets there near the end of training, and only without dropout. The MLP never
fully does, which goes some way to explaining why it is the one model where dropout's accuracy
effect is measurable.

## The loss plot: where dropout shows up

Accuracy only checks whether the right class has the highest score. Cross-entropy also checks how
much probability the model put on it, and it punishes confident mistakes hard.
{numref}`fig-do-curves` tracks the validation loss through training.

```{figure} figures/val-loss-curves.png
:alt: Validation loss per epoch, three panels, one line per dropout rate. In the MLP panel p=0.75 is highest throughout. In the CNN panel the p=0 line levels off and ends above the other three. The ResNet-18 panel is noisy, with the p=0.75 line highest and spiking near epoch 29.
:name: fig-do-curves

Validation loss per epoch, mean of three seeds. Darker lines are higher dropout rates. In the
CNN panel, the lightest line ($p = 0$) stops improving in the second half of training and ends
above every dropout setting, the usual signature of overfitting. Note that the three panels have
different y-axes.
```

| Dropout p | MLP | CNN | ResNet-18 |
|---:|:---:|:---:|:---:|
| 0 | 0.109 | 0.087 | 0.068 |
| 0.25 | **0.099** | 0.059 | **0.066** |
| 0.5 | 0.100 | **0.054** | 0.067 |
| 0.75 | 0.130 | 0.060 | 0.093 |

*Final validation loss (cross-entropy), mean of three seeds.*

For the CNN this is the largest effect in the experiment. Dropout at $p = 0.5$ lowers the final
validation loss by 38%, from 0.087 to 0.054, while validation accuracy doesn't move (98.4% in both
cases). Without dropout the CNN's validation loss bottoms out at 0.081 at epoch 24 and ends at
0.087. Loss rising while accuracy holds is the usual sign of a network that keeps getting the same
images right while growing more confident about the ones it gets wrong. With dropout in its dense
head, the loss settles about a third lower and stays there. The MLP shows the same effect at a smaller
size (−9% at $p = 0.25$). ResNet-18 shows none: its loss is flat across $p \le 0.5$, and at
$p = 0.75$ it is clearly worse and noisy, bottoming out at epoch 19 and ending 27% above its
minimum.

So on a problem this easy, dropout's useful contribution is to the *quality of the probabilities*
rather than to the decisions. Whether that matters depends on what the model is for. A classifier
whose scores feed a threshold, a ranking or a downstream decision benefits from it. A leaderboard
that reports top-1 accuracy will not notice.

## The test set

The best rate per model, chosen on validation and then scored once on the 2,007-image test set:

| Model | Best p | Train acc. | Validation acc. | Test acc. |
|---|---:|---:|---:|---:|
| MLP | 0.25 | 99.6 | 97.6 | 94.0 |
| CNN | 0.25 | 99.9 | 98.5 | 95.7 |
| ResNet-18 | 0.5 | 100.0 | 98.8 | 96.6 |

*Accuracy (%), mean of three seeds.*

Two observations. First, test accuracy is 2 to 3.6 points below validation at every dropout rate,
for every model. That drop is larger than anything dropout did. USPS's official test set is a
separate collection from the training pool the validation images were drawn from, and the gap
measures that difference, not overfitting. Second, the test set tells a slightly different story
about who benefits. Across $p$, the MLP's test accuracy moves by 0.6 points and the CNN's by 0.2.
ResNet-18 goes from 95.9% at $p = 0$ to 96.6% at $p = 0.5$, the largest test gain in the experiment
and the opposite of prediction 3. With three seeds and a 0.4-point spread at $p = 0$, I would call
that a hint rather than a result. It is about 14 test images.

## Scorecard

| Prediction | Verdict |
|---|---|
| 1. The MLP gains the most from dropout | **Confirmed on validation**, but small: +0.5 points at $p = 0.25$. On test, ResNet-18's gain is larger. |
| 2. The gap shrinks as $p$ grows, for MLP and CNN | **Confirmed.** MLP 2.6 → 1.8, CNN 1.6 → 0.9 points. At $p = 0.75$ it shrinks by losing training fit. |
| 3. ResNet-18 barely reacts | **Confirmed for validation accuracy** (0.4-point range), not for loss: $p = 0.75$ raises its validation loss 36% over $p = 0$. |
| 4. At $p = 0.75$ MLP and CNN start to underfit | **Weakly.** Training accuracy drops (MLP 99.8 → 98.4, CNN 100.0 → 99.1) and validation drops with it, but both still fit their training data near-perfectly. |

## So, is dropout worth it?

On a dataset that is small but easy, dropout is not the lever that moves accuracy. All three
models generalised to within three points of their training accuracy without it, and the best
dropout rate bought at most half a point. What dropout did change was confidence. In the CNN, noise
in the dense head kept the validation loss from creeping up late in training and cut it by more
than a third, with the decisions untouched. Where the noise was injected mattered as much as how
much. Dropout spread across the MLP slowed its fit. Dropout in front of ResNet-18's classifier
delayed its memorisation by a few epochs, and at $p = 0.75$ mainly made its outputs noisier.

The broader lesson is about measurement. The accuracy plot, the gap plot and the loss plot each
told a different story, and the one that matters depends on what the model is for. Three seeds, a
1,000-image validation set and differences measured in single images also put a floor under what
this experiment can resolve. A follow-up worth running is the same sweep with augmentation or a
harder dataset, where there is real overfitting for dropout to fight.

[^srivastava]: N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov,
    "Dropout: A Simple Way to Prevent Neural Networks from Overfitting," *JMLR* 15, 2014.
[^li]: X. Li, S. Chen, X. Hu, J. Yang, "Understanding the Disharmony between Dropout and Batch
    Normalization by Variance Shift," CVPR 2019.
[^lecun]: Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, L. D. Jackel,
    "Backpropagation Applied to Handwritten Zip Code Recognition," *Neural Computation* 1(4), 1989.
[^usps]: J. J. Hull, "A Database for Handwritten Text Recognition Research," *IEEE TPAMI* 16(5),
    1994.
