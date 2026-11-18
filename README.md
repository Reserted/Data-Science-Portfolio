# Data Science Portfolio

A [Jupyter Book 2](https://next.jupyterbook.org) (MyST) portfolio site. Notebooks execute
on every deploy, so the charts and numbers on the site are always the ones the code
actually produces.

```
├── myst.yml                     # site config: title, authors, nav, table of contents
├── index.md                     # landing page
├── about.md                     # bio, experience, skills
├── projects.md                  # project gallery
├── projects/
│   ├── customer-churn.ipynb     # executable notebook project
│   └── sales-forecasting.md     # written case-study template
├── lab.md                       # coursework / exercises index
├── lab/
│   ├── linear-regression.ipynb  # least squares from the normal equations
│   ├── B - …, C - …, E1 - …     # deep-learning lab notebooks (Lab Tasks 2–6)
│   ├── Pokemon Classification.ipynb  # lab exercise: model inference on a custom dataset
│   └── figures/                 # diagrams the lab notebooks embed
├── blog.md                      # blog index
├── blog/                        # one folder per post
│   ├── learning-rate/           # also batch-size/, dropout/, weight-decay/ (draft: not in toc)
│   │   ├── learning-rate.md          # the post
│   │   ├── learning-rate-colab.ipynb # the Colab notebook that produced every number in it
│   │   └── figures/                  # its output: PNGs, results.md, runs.csv
│   ├── jupyter-book-site/       # post + screenshots
│   ├── diagrams/                # diagrams.py + the SVGs it writes, shared by the posts
│   └── archive/                 # superseded posts, kept out of the site
├── requirements-dl.txt          # PyTorch / Lightning env for the lab and blog notebooks
├── style.css                    # the entire visual identity — see below
├── data/
│   ├── make_sample_data.py      # regenerates the sample dataset
│   └── subscribers.csv
├── requirements.txt
└── .github/workflows/deploy.yml # GitHub Pages deploy
```

## Run it locally

```bash
pip install -r requirements.txt
jupyter book start          # live preview at http://localhost:3000, reloads on save
```

Build the static site the way CI does:

```bash
jupyter book build --html --execute     # output lands in _build/html/
```

## The look

`style.css` is the whole theme. MyST loads it last, after the `book-theme` stylesheet,
so plain element selectors win and `!important` is only needed where a Tailwind utility
class (`.bg-white`, `.text-slate-*`) outranks one.

- Colours are two token sets — `:root` for light, `html.dark` for dark. Change those
  ~7 variables and the entire site changes; nothing else hard-codes a colour.
- Fonts come from one Google Fonts `@import`: Instrument Serif (display), Inter Tight
  (body), JetBrains Mono (code and all small-caps labels).
- Section numbers (`01`, `02`, …) are a CSS counter on `.article h2`, not written into
  the Markdown.
- Sections, figures, cards and tables fade in on scroll via `animation-timeline: view()`
  — no JavaScript. It is wrapped in `@supports` so browsers without it simply show the
  content, and in `prefers-reduced-motion: no-preference` so it is skipped for anyone
  who asked for less motion.

## Live code

`project.thebe.lite: true` in `myst.yml` boots a Pyodide kernel in the visitor's browser,
so every code cell on the site is editable and runnable with no server and no Binder
round-trip. Only packages Pyodide ships (numpy, pandas, matplotlib, scikit-learn) will
import — a notebook needing anything else still renders its committed outputs fine.

## Make it yours

1. **`myst.yml`** — set `title`, `authors`, `description`, `site.title`, `site.actions`,
   and uncomment `github:` with your repo URL. This file also holds the `toc:` — every
   page must be listed there to appear in the sidebar.
2. **`index.md` / `about.md`** — replace the placeholder copy. Search for
   `your-username`, `you@example.com`, and `Your Name` to catch every stub:
   ```bash
   grep -rn "your-username\|you@example.com\|Your Name" --include=*.md --include=*.yml .
   ```
3. **Add a project** — drop a `.ipynb` or `.md` into `projects/`, add it under
   `projects` in the `toc:`, and add a `{card}` to `projects/index.md`.
4. **Swap the data** — `data/subscribers.csv` is synthetic. Point the notebook at real
   data and delete `make_sample_data.py`.

### Notebook conventions

- Give each notebook front matter (`title`, `subtitle`, `date`, `tags`) in its
  notebook-level metadata — Jupyter Book renders it as the page header.
- **Paths**: the book build executes notebooks from the project root, while JupyterLab
  runs them from the notebook's folder. The churn notebook resolves both with a
  `DATA = Path("data") if Path("data").is_dir() else Path("../data")` line — copy it.
- Tag a cell `remove-input` to hide code but keep the chart, or `remove-cell` to drop it
  from the site entirely.

## Deploy

### GitHub Pages (workflow included)

1. Push to a GitHub repo on the `main` branch.
2. **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. Every push to `main` rebuilds and publishes to
   `https://<user>.github.io/<repo>/`.

The workflow sets `BASE_URL` to the repo name so asset paths resolve in the subdirectory.
If you deploy to a custom domain or a `<user>.github.io` root repo, remove that `env:` block.

### Netlify / Vercel / Cloudflare Pages

| Setting | Value |
|---|---|
| Build command | `pip install -r requirements.txt && jupyter book build --html --execute` |
| Publish directory | `_build/html` |
| Python version | `3.12` |

Leave `BASE_URL` unset on these — they serve from the domain root.

## Notes

- Builds are reproducible: the sample data uses a fixed seed, so committed and rebuilt
  numbers match.
- `_build/` is gitignored — the deploy builds from source.
- If a notebook is slow or needs credentials, commit it with outputs and drop
  `--execute` from the build command.
- The deep-learning notebooks (in `lab/` and `blog/`) need PyTorch and a GPU, which CI
  does not have. Each carries `execute: {skip: true}` in its notebook metadata, so the
  build renders the committed outputs instead of re-running them.
- To regenerate a blog post's results, open its `*-colab.ipynb` in Colab on a T4 GPU and
  run all. The last cell downloads `<slug>.zip`; unzip it inside `blog/` and the files land
  in `blog/<slug>/figures/`. After editing the model code, rerun
  `python blog/diagrams/diagrams.py` to redraw the architecture diagrams.
- Lab notebooks run locally:
  ```bash
  pip install -r requirements-dl.txt
  jupyter nbconvert --to notebook --execute --inplace "lab/E1 - CNN Implementation.ipynb"
  ```
  The Pokémon notebook expects the Kaggle dataset at `lab/data/archive/PokemonData`
  (gitignored).
