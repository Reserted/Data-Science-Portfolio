---
title: MLBB Draft Win Predictor
description: Predicting competitive Mobile Legends match outcomes from pre-game draft information alone.
---

**Stack:** Python, scikit-learn, Jupyter.

## The question

In competitive Mobile Legends, the draft happens before anyone touches a keyboard:
each team picks and bans heroes, and one side picks first. How much of the match
outcome is already decided by then?

That framing is what makes the project honest. No in-game data, no gold graphs, no
first-blood timers — only what is known before the match starts. Everything else
would be leakage.

## Pipeline

Built end to end in Jupyter:

1. **Ingestion** — Liquipedia API, pulling match results with their full draft.
2. **Cleaning** — normalising hero names, handling incomplete drafts, deduplicating.
3. **Feature engineering** — hero picks and bans, side selection, and derived
   team-composition features.
4. **Modelling** — tuned classifiers, judged against the **proportional chance
   criterion** rather than raw accuracy.
5. **Interpretation** — feature-importance analysis on what the draft actually
   contributes.

## Why the proportional chance criterion

A two-class problem with an uneven split makes accuracy flattering: predict the
majority side every time and the number still looks respectable. The proportional
chance criterion sets the bar at what random guessing that respects the class
proportions would achieve, so a model only counts as informative when it clears that.

## Notes

Write-up in progress — model results and feature-importance figures to follow.
