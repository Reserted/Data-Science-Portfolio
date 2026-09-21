---
title: Regional Sales Forecasting
subtitle: Twelve-week demand forecasts across 40 regions
date: 2026-01-15
tags: [time series, forecasting, case study]
---

:::{admonition} Case study template
:class: tip dropdown
This page is the structure to reuse when the code can't be published. Keep the five
headings; replace every paragraph. A reader should get the whole story in 90 seconds.
:::

## The problem

Two or three sentences on the business situation. Who was hurting, and how much did
it cost them? Name the decision the model was meant to support — not the algorithm.

> Planners were ordering against a spreadsheet extrapolation, leaving ~$2.1M a year
> in stockouts and markdowns across 40 regions.

## The data

- **Sources** — what tables, what systems, what date range.
- **Grain** — one row per _region × week_, 40 regions × 156 weeks.
- **The messy bit** — the thing that took the longest. Missing weeks, a definition
  change halfway through history, a promo calendar nobody had digitised.

## The approach

What you tried, in order, and why you stopped where you did.

1. **Baseline** — seasonal naive. Every model has to beat this or it doesn't ship.
2. **Model** — the approach you landed on, and the one honest sentence on why.
3. **Validation** — rolling-origin backtest, 12-week horizon, MAPE and pinball loss.

| Model | MAPE | vs. baseline |
|---|---|---|
| Seasonal naive | 21.4% | — |
| Gradient boosting + calendar features | 13.9% | **−35%** |
| Ensemble | 13.1% | −39% |

## The outcome

The number that mattered, and who is using it now. Include what you'd do differently —
that reads as experience, not weakness.

- Forecast error down 35%; planners adopted it for 34 of 40 regions.
- Weekly accuracy review kept trust alive after launch.
- **Next time**: build the monitoring dashboard in week one, not month three.

## Links

- [Code](https://github.com/your-username/your-repo) (if public)
- [Write-up / slides](https://example.com)
