---
title: Lab
subtitle: Coursework, exercises, and things tried once to see what happened
---

This is the working half of the site. Where the project pages present finished work,
these are the notebooks behind it — coursework exercises, method checks, and small
experiments kept in whatever state they ended in. The NumPy and pandas notebooks execute
on every deploy, so their numbers and charts are the ones the code produced on the last
build. The deep-learning notebooks need PyTorch and a GPU, so they are run locally and
published with the outputs of that run. Nothing here is polished, and that is the point:
it is the record of the learning rather than the advertisement for it.

::::{grid} 1 1 2 2

:::{card} Least Squares by Hand
:link: lab/linear-regression.ipynb
:header: Notebook

Linear regression solved from the normal equations with NumPy alone, cross-checked
against `polyfit`, then read through its residuals.
+++
`numpy` · `regression`
:::

:::{card} Customer Churn — Prediction & Drivers
:link: projects/customer-churn.ipynb
:header: Notebook

Ranking at-risk subscribers and isolating what drives them away.
+++
`pandas` · `classification`
:::

::::

## Deep learning

::::{grid} 1 1 2 2

:::{card} Understanding Deep Learning
:link: lab/B - Understanding Deep Learning.ipynb
:header: Lab Tasks 2–4

A ReLU forward pass by hand, backpropagation in NumPy checked against autograd, and a
two-layer regression net trained with SGD for 1,000 epochs.
+++
`numpy` · `pytorch` · `backprop`
:::

:::{card} PyTorch Basics
:link: lab/C - PyTorch Basics.ipynb
:header: Lab Task 5

Seeding, dtypes, reshaping, slicing, and why `x.pow(2)` leaves `x` alone.
+++
`pytorch` · `tensors`
:::

:::{card} CNN Implementation
:link: lab/E1 - CNN Implementation.ipynb
:header: Lab Task 6

An MNIST CNN taken apart layer by layer, then a deeper CNN built from an architecture
diagram, shape-checked and trained.
+++
`pytorch` · `cnn`
:::

:::{card} Model Inference on Custom Dataset
:link: lab/Pokemon Classification.ipynb
:header: Lab Exercise

ResNet-50 fine-tuned on 150 Pokémon classes with Lightning, then loaded from its
checkpoint to predict on images it never saw.
+++
`lightning` · `transfer learning`
:::

::::
