---
title: How I Built This Site
subtitle: A Jupyter Book 2 portfolio from an empty folder to GitHub Pages. Also a list of every way the theme fought back.
date: 2026-11-18
authors:
  - name: Eleijahmar Fifth C. Sevare
tags: [jupyter book, myst, github pages, css, tutorial]
---

You are reading a website about how this website was made. Yes, that's recursive. No, I won't do
another post about writing this post.

This site is a [Jupyter Book 2](https://next.jupyterbook.org) project. Every page is a Markdown
file or a Jupyter notebook, one YAML file wires them together, and a GitHub Action builds and
publishes the whole thing every time I push. Below are the real files from the repo, in roughly
the order I wrote them and broke them.

```{figure} figures/home.png
:name: fig-site-home
:alt: The portfolio home page: "sevare" logo, a top nav with Projects, Lab, Blog and About, a sidebar table of contents, the page title in a display serif, and three featured-work cards.

The finished home page. Everything on it comes from three files: `index.md`, `myst.yml` and
`style.css`. Everything *wrong* with it at various points also came from those three files.
```

## 1. Install and preview

Jupyter Book 2 is built on [MyST](https://mystmd.org). If you remember version 1, forget it:
there's no Sphinx anymore. It's a Node.js site generator wearing a Python package as a disguise,
so you install it with `pip`, and then it quietly needs Node.js on your machine as well.

`requirements.txt` holds only what the site and its runnable notebooks need:

```{code} text
:filename: requirements.txt
jupyter-book>=2.1,<3
ipykernel>=6.29
numpy>=2.0
pandas>=2.2
matplotlib>=3.9
```

```bash
pip install -r requirements.txt
jupyter book init     # writes a starter myst.yml (only needed once)
jupyter book start    # live preview at http://localhost:3000, reloads on save
```

`jupyter book start` is where I lived for most of this project. Save a file and the page reloads
a second later, which is the perfect speed for trying forty shades of grey-blue in a row.

## 2. The repo layout

```text
├── myst.yml          # the brain: nav, table of contents, theme options
├── index.md          # home page
├── about.md          # me, but in Markdown
├── projects.md       # gallery page + projects/ folder
├── lab.md            # coursework index + lab/ notebooks
├── blog.md           # you are here-ish
├── style.css         # the entire personality
├── public/           # a font and one stubborn button
├── requirements.txt
└── .github/workflows/deploy.yml   # the robot that publishes it
```

Each section is an index page plus a folder of children. `lab/` is mostly `.ipynb`, `blog/` is
mostly `.md`, and Jupyter Book genuinely does not care which is which. Treating them equally is
the nicest thing about it.

## 3. `myst.yml`: the one file to rule them all

The config has two halves. `project:` is *what* the site contains. `site:` is *how* it looks.

```{code} yaml
:filename: myst.yml (project)
version: 1
project:
  title: Eleijah — Data Science Portfolio
  github: https://github.com/Reserted/Data-Science-Portfolio
  license: CC-BY-4.0
  # In-browser Python: every code cell on the site is runnable, no server.
  thebe:
    lite: true
  toc:
    - file: index.md
    - file: about.md
    - file: projects.md
      children:
        - file: projects/respira.md
        - file: projects/customer-churn.ipynb
    - file: lab.md
      children:
        - file: lab/linear-regression.ipynb
        - file: lab/C - PyTorch Basics.ipynb
    - file: blog.md
      children:
        - file: blog/learning-rate/learning-rate.md
          children:
            - file: blog/learning-rate/learning-rate-colab.ipynb
```

`toc:` *is* the sidebar. **If a file isn't listed, it doesn't show up in the navigation.** That
sounds like a gotcha, and it is a feature: my half-finished weight-decay post sits in `blog/`
right now, invisible, like a draft email I'm too scared to send. Nesting `children` under a post
is how each blog post carries its Colab notebook underneath it.

```{code} yaml
:filename: myst.yml (site)
site:
  template: book-theme
  title: Eleijah — Portfolio
  nav:
    - title: Projects
      url: /projects
    - title: Lab
      url: /lab
    - title: Blog
      url: /blog
    - title: About
      url: /about
  actions:
    - title: LinkedIn
      url: https://linkedin.com/in/eleijah-sevare
  options:
    logo_text: sevare
    style: style.css
```

`nav` is the row of links across the top. `actions` are the outlined buttons on the right (hi,
recruiters). `options.style` loads my stylesheet *after* the theme's, which matters a lot in
section 6.

## 4. Writing pages

A page is Markdown with a YAML hat on. `title` and `subtitle` become the giant heading.
`date`, `authors` and `tags` show up on blog posts so everyone knows exactly who to blame:

```{code} markdown
:filename: blog/dropout/dropout.md
---
title: Dropout Changes Confidence, Not Accuracy
subtitle: One hyperparameter, three architectures, USPS digits.
date: 2026-11-18
authors:
  - name: Eleijahmar Fifth C. Sevare
tags: [deep learning, dropout, regularization, pytorch, usps]
---
```

The index pages are built from MyST **directives**: fenced blocks with a name in curly braces. The
whole Blog page is one `grid` full of `card`s:

```{code} markdown
:filename: blog.md
::::{grid} 1 1 3 3

:::{card} Where Learning Rates Break
:link: blog/learning-rate/learning-rate.md
:header: November 2026

All three models peak at the same rate. Kuzushiji-MNIST, 54 runs.
+++
`learning rate` · `kmnist`
:::

::::
```

Decoding the hieroglyphs:

- `1 1 3 3` is the number of columns on phone, small tablet, tablet and laptop. One card per row
  on your phone, three across on a laptop.
- The grid's fence has **four** colons and each card has **three**. That's how MyST tells which
  thing lives inside which. Get it wrong and you get a very confident, very broken page. Count
  your colons. I'm serious.
- Everything after `+++` goes into the card's footer, which is where the tag chips live.

```{figure} figures/blog.png
:name: fig-site-blog
:alt: The Blog index page: title "Blog", a one-paragraph intro, and three cards in a row, each with a "November 2026" header, a title, a summary and tag chips.

`blog.md`, rendered. The sidebar tree on the left is the `toc:` from `myst.yml`, doing its job.
```

Figures, equations and cross-references are directives too. Give a figure a `:name:`, and
`` {numref}`fig-site-blog` `` anywhere on the site turns into a numbered link to it. This post
uses the same syntax for its screenshots, which means this page is now documenting itself. We
promised not to go further down that hole.

## 5. Notebooks as pages

This is the good part. A `.ipynb` in the toc renders like any other page: markdown cells become
prose, code cells show their source *and* output, and the first `# Heading` becomes the title.
Build with `--execute` (section 8) and every notebook runs fresh, so **the chart on the page is
the chart the code made on that build**. Nobody can accuse me of photoshopping my loss curves.

The catch: some notebooks can't run on the build server. The deep-learning ones want PyTorch and
a GPU, and GitHub's free runner has the computing ambitions of a potato. Those notebooks carry
one key in their metadata:

```{code} json
:filename: blog/dropout/dropout-colab.ipynb (metadata)
{
  "execute": { "skip": true },
  "accelerator": "GPU"
}
```

`skip` means "don't run me, just show what I already have." So I run those on Colab (which *does*
lend you a GPU), save, and commit them with the outputs inside.

And then there's the magic trick. `thebe: lite: true` in `myst.yml` puts a little power button on
every notebook page. Press it and a [Pyodide](https://pyodide.org) Python kernel boots **inside
your browser**. Every code cell becomes editable and runnable. No server, no Binder, no "please
wait while we spin up your container", no money spent by me. The only limit is that it can import
what Pyodide ships (NumPy, pandas, matplotlib, scikit-learn), so don't go looking for PyTorch in
there.

```{figure} figures/lab.png
:name: fig-site-lab
:alt: The "Least Squares by Hand" lab notebook page, with a power button at the top right of the content and a code cell showing NumPy and matplotlib imports.

A lab notebook as a page. The power button at the top right starts Python in your browser. Go on,
press it. Break something. It's your browser, not mine.
```

## 6. Styling: one CSS file versus the theme

The default `book-theme` is perfectly nice, in the way a hotel room is perfectly nice. All of the
personality here lives in `style.css`, which MyST loads **after** the theme so my rules usually
win the argument.

The whole palette is seven variables, written once for light and once for dark:

```{code} css
:filename: style.css
:root {
  --font-display: 'Rene', Georgia, serif;
  --font-sans: system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', ui-monospace, monospace;

  --bg:      #EEF1F4;
  --bg-2:    #E1E6EC;
  --fg:      #11161D;
  --fg-dim:  #57616D;
  --rule:    #CBD3DC;
  --accent:  #3457D5;
}

html.dark {
  --bg:      #0A0D12;
  --bg-2:    #121721;
  --fg:      #DBE1E8;
  --fg-dim:  #7C8794;
  --rule:    #232B37;
  --accent:  #6E8FFF;
}
```

The theme's sun/moon button just slaps a `dark` class on `<html>`. Redefine the variables under
`html.dark` and that's the entire dark mode. No rule further down hard-codes a colour, so swapping
those seven lines reskins the whole site. It's the cheapest makeover in web development.

:::::{grid} 1 1 2 2

::::{figure} figures/home.png
:alt: Home page in light mode.

Light mode, for people who go outside.
::::

::::{figure} figures/home-dark.png
:alt: The same home page in dark mode: near-black background, light text, same layout.

Dark mode, for people who code at 2 a.m. (me).
::::

:::::

After that I went through the theme one component at a time: `.myst-card`,
`.myst-fm-block-title`, `.myst-toc-item`, `.myst-admonition`. I found every class name with
right-click → **Inspect**, which is the real name of this whole tutorial. Wherever the theme uses
a Tailwind utility class (`.bg-white`, `.text-slate-*`), it outranks a normal selector, and
*only there* do you need to bring out `!important`:

```{code} css
:filename: style.css
.myst-card {
  background: transparent !important;
  border: 1px solid var(--rule) !important;
  border-radius: 0 !important;
  box-shadow: none !important;
}
/* Accent bar that grows down the left edge on hover, only on cards that are links. */
a.myst-card::after {
  content: '';
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 2px;
  background: var(--accent);
  transform: scaleY(0);
  transform-origin: top;
  transition: transform .3s cubic-bezier(.2,.8,.2,1);
}
a.myst-card:hover::after { transform: scaleY(1); }
```

Hover over a card on the [Blog page](../../blog.md). That little blue bar sliding down? Twelve lines
of CSS. Worth it.

### A custom font

The fancy display face is Rene, a local `.woff2` in `public/fonts/`, loaded with `@font-face`:

```{code} css
:filename: style.css
@font-face {
  font-family: 'Rene';
  src: local('ReneRegular'), local('Rene-Regular'),
       url('./fonts/Rene-Regular.woff2') format('woff2');
  font-display: swap;
}
```

Looks done, right? It is not done. MyST copies `style.css` into the build and *leaves the font
behind*, like a friend who brings the chips but forgets the dip. Section 8 goes back for it.

## 7. A navbar button with an anywidget

I wanted a **Plain** button: one click turns off the display serif and the grid texture for anyone
who'd rather read than admire typography. Simple, right? Just add a `<script>` tag.

MyST strips `<script>` tags. It strips raw HTML in that slot too. The *officially supported* way
to run your own JavaScript is an [anywidget](https://anywidget.dev) module, which is a bit like
being told the only way into your house is through the chimney. But it works, and
`parts.navbar_end` drops it into the top bar on every page:

```{code} yaml
:filename: myst.yml
site:
  parts:
    navbar_end: |
      :::{anywidget} public/plain-toggle.mjs
      :::
```

The module exports a `render` function that gets an element to draw into. All the button does is
toggle a `plain` class on `<html>` and remember your choice:

```{code} javascript
:filename: public/plain-toggle.mjs
export default {
  render({ el }) {
    const root = document.documentElement;
    const button = document.createElement('button');
    button.className = 'plain-toggle';
    button.textContent = 'Plain';

    const paint = () =>
      button.setAttribute('aria-pressed', String(root.classList.contains('plain')));

    button.addEventListener('click', () => {
      const on = !root.classList.contains('plain');
      root.classList.toggle('plain', on);
      try { localStorage.setItem('plain', on ? '1' : '0'); } catch { /* private mode */ }
      paint();
    });

    try { if (localStorage.getItem('plain') === '1') root.classList.add('plain'); } catch {}
    paint();

    // Client-side navigation and the dark-mode switch both rewrite <html>'s classes.
    const observer = new MutationObserver(paint);
    observer.observe(root, { attributes: true, attributeFilter: ['class'] });

    el.append(button);
    return () => observer.disconnect();
  },
};
```

Then `style.css` does the rest with the same trick as dark mode, re-pointing the variables:

```{code} css
:filename: style.css
html.plain {
  --font-display: var(--font-sans);
  --font-accent:  var(--font-sans);
  --grain: transparent;
}
html.plain body { background-image: none; }
```

Plot twist: anywidgets render inside a **shadow root**, a sealed little bubble that `style.css`
can't reach. So the button can't be styled from the stylesheet. Its CSS lives as a string inside
the `.mjs` file and gets glued on next to it. It's a button that brings its own outfit.

Go ahead and press **Plain** at the top of this page. I won't be offended. (I will be a little
offended.)

## 8. Deploying to GitHub Pages

Now for the robot. The workflow installs the requirements, builds the HTML with every notebook
executed, and hands `_build/html` to GitHub Pages:

```{code} yaml
:filename: .github/workflows/deploy.yml
on:
  push:
    branches: [main]

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: pip
      - run: pip install -r requirements.txt

      - name: Build site
        env:
          BASE_URL: /${{ github.event.repository.name }}
        run: jupyter book build --html --execute

      - name: Copy the display font next to the stylesheet
        run: mkdir -p _build/html/build/fonts && cp public/fonts/Rene-Regular.woff2 _build/html/build/fonts/

      - uses: actions/upload-pages-artifact@v3
        with:
          path: ./_build/html

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

Most of that is copy-paste boilerplate. Three lines are not, and each one is a trap:

- **`BASE_URL`.** A project site lives at `username.github.io/<repo>/`, not at the domain root.
  Leave this out and every link and stylesheet points at `/`. The site loads as raw, unstyled
  HTML, like it's 1996 and you've just discovered `<blink>`.
- **The font copy.** Here's where we go back for the dip. The stylesheet gets built as
  `build/style-<hash>.css` and looks for `./fonts/Rene-Regular.woff2` next to itself. MyST doesn't
  put it there, so this step does.
- **`--execute`.** This is what makes the notebook outputs fresh on every deploy. Notebooks
  marked `skip` are left alone with their committed outputs.

Then, on GitHub: **Settings → Pages → Source: GitHub Actions**. From now on every push to `main`
redeploys the site in a couple of minutes, giving you just enough time to spot the typo you just
published.

Want to see the production build before the whole internet does?

```bash
jupyter book build --html --execute
python -m http.server -d _build/html 8000
```

That's also how I took the screenshots in this post: headless Chrome pointed at that local server.
Chrome took pictures of my website so my website could show you pictures of my website.
(Okay, that's the last recursion joke. Probably.)

## 9. Things the theme fought me on

Every one of these is a comment in `style.css` now, left there like warning signs on a hiking
trail:

- **`@import` must be the very first rule.** The Google Fonts import for JetBrains Mono has to sit
  above the `@font-face` block. Put it anywhere else and the browser doesn't complain. It just
  silently ignores it. Ghosted by a stylesheet.
- **Headings ignore `margin-top`.** The theme sets `.article-grid > * { margin-top: 0 !important }`
  on everything, so headings get their breathing room from `padding` instead. The `!important` was
  already taken.
- **Never give `.myst-primary-sidebar` a background.** On wide screens it is secretly a
  full-width invisible overlay. Paint it, and an opaque sheet appears over the entire article. The
  sidebar ate the website.
- **Animate `translate`, not `transform`.** A fade-in animation on `transform` locked it to `none`
  and killed the hover lift on the cards. Two animations entered, one left.
- **Scroll-driven animations play backwards.** `animation-timeline: view()` looks great scrolling
  down, then rewinds when you scroll up, which reads as flickering. Replaced with a single fade-in
  on load. Less fancy, much less haunted.

## Takeaway

The whole site is a folder of Markdown and notebooks, one YAML config, one CSS file and one
workflow. The best part isn't the fonts or the hover bars. It's that **notebooks are pages**: the
lab work and the blog experiments are published as the same files that made the results, not as
screenshots of them. (Except in this post. This post is all screenshots. I contain multitudes.)

The source is on [GitHub](https://github.com/Reserted/Data-Science-Portfolio), and the edit icon at
the top of every page takes you to that page's file. Steal whatever you like. Just count your
colons.
