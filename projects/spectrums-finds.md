---
title: Spectrum's Finds — YouTube Niche Research Tool
description: Outlier scoring and LLM title clustering to find what packaging actually drives views in a niche.
---

**Stack:** Python, yt-dlp, LLM APIs.

## The problem

Raw view counts tell you which channels are big, not which videos worked. A 200k-view
video from a channel that averages 500k underperformed; the same 200k from a channel
that averages 10k is the signal you want.

## How it works

**Harvest.** Hundreds of videos per niche via `yt-dlp`, fetched by a background
worker with cursor pagination, deduplication, and backoff.

**Score outliers.** Each video's views are measured against **its own channel's
median**, not against the niche. That normalisation is the whole point: it strips out
channel size and audience, leaving what the title and packaging contributed.

**Cluster formats.** Winning titles go to an LLM, which groups them into reusable
title formats rather than one-off examples.

**Keep the work.** Persistent research spaces per niche, with JSON and CSV export.

## Notes

Write-up in progress — example niche breakdown to follow.
