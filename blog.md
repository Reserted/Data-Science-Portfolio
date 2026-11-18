---
title: Blog
subtitle: Deep-learning experiments, written up for classmates
---

Each post changes **one** hyperparameter and trains the same three models on it: an MLP, a small
CNN and ResNet-18. Each uses a less common `torchvision` dataset, writes its predictions down
before the runs, and links the Colab notebook that produces every figure and number in it.

::::{grid} 1 1 3 3

:::{card} Where Learning Rates Break
:link: blog/learning-rate/learning-rate.md
:header: November 2026

At 0.3, two of three networks died with a loss of exactly ln 10. ResNet-18 shrank its own step
size and lived. Cursive from pre-modern Japanese books, 54 runs.
+++
`learning rate` · `kmnist`
:::

:::{card} Batch Size Versus Training Time
:link: blog/batch-size/batch-size.md
:header: November 2026

Which batch size wins depends on the clock: epochs, updates or seconds. Per update, the bigger
batch was always ahead. EuroSAT satellite patches, 36 runs.
+++
`batch size` · `eurosat`
:::

:::{card} Dropout Changes Confidence, Not Accuracy
:link: blog/dropout/dropout.md
:header: November 2026

Picked to hand dropout an easy win. Accuracy moved a point at most, while validation loss dropped
by over a third. ZIP codes from 1980s US mail, 36 runs.
+++
`dropout` · `usps`
:::

::::

## About this site

::::{grid} 1 1 3 3

:::{card} How I Built This Site
:link: blog/jupyter-book-site/jupyter-book-site.md
:header: November 2026

A website about how this website was made. `myst.yml`, card grids, notebooks as pages, one CSS
file versus the theme, and the deploy robot.
+++
`jupyter book` · `myst` · `css`
:::

::::
