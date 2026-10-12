# Snakamoto — snakamoto404.github.io

Astro static site scaffolded for GitHub Pages user-site hosting.

Live site: https://snakamoto404.github.io/

## Features

- Home page at `/`
- Archive index at `/archive/`
- Daily archive pages at `/archive/YYYY-MM-DD/` from markdown content collection
- Subscribe and unsubscribe form pages at `/subscribe/` and `/unsubscribe/`
- Giscus comments component on each daily archive page
- LaTeX/math rendering in Markdown (remark-math + rehype-katex)
- GitHub Actions workflow for Pages deployment

## Reproducible environment

- Node is pinned to **v20** via `.nvmrc` / `.node-version` (matches GitHub Pages CI).
- Dependencies are locked via `package-lock.json`; prefer `npm ci` for deterministic installs.

## Local setup

1. Install Node 20 (pick one):

```bash
nvm use
# or: fnm use
```

2. Install dependencies (deterministic):

```bash
npm ci
```

3. Run dev server:

```bash
npm run dev
```

4. Build for production:

```bash
npm run build
```

## VS Code writing workflow

This local checkout includes Node 20.20.2 in ignored `.tools/node/`. New VS Code
terminals automatically use it; dependencies are already installed with `npm ci`.
The Astro extension provides `.astro` language support.

- Run **Tasks: Run Task → Dev: live preview**, then open http://localhost:4321.
  Saving Markdown updates the browser automatically. Stop the task with Ctrl+C.
- Press **Cmd+Shift+B** for a production build; generated pages are in `dist/`.
- Write posts in `src/content/blogs/<topic>/`. The μP scaffold demonstrates
  `:::theorem[Title]{#label}`, `:ref[label]`, math, and Markdown reference links.
- Deprecated posts are excluded from both dev and production pages by
  `DEPRECATED_BLOG_SLUGS` in `src/utils/blogVisibility.ts`; their source is retained.

For a terminal outside VS Code, run `export PATH="$PWD/.tools/node/bin:$PATH"`
from this checkout first. Other clones can use the standard Node setup above.

## Add a new daily archive entry

Create a markdown file in `src/content/archive/` named `YYYY-MM-DD.md`:

```md
---
title: Daily Feed - YYYY-MM-DD
date: YYYY-MM-DD
summary: One-line summary.
---

Your daily content here.
```

The page is generated automatically at `/archive/YYYY-MM-DD/`.

## Giscus placeholders

Set these public env vars (for local `.env` or GitHub repo variables):

- `PUBLIC_GISCUS_REPO`
- `PUBLIC_GISCUS_REPO_ID`
- `PUBLIC_GISCUS_CATEGORY`
- `PUBLIC_GISCUS_CATEGORY_ID`
- `PUBLIC_GISCUS_MAPPING` (default: `pathname`)
- `PUBLIC_GISCUS_REACTIONS_ENABLED` (default: `1`)
- `PUBLIC_GISCUS_EMIT_METADATA` (default: `0`)
- `PUBLIC_GISCUS_THEME` (default: `preferred_color_scheme`)
- `PUBLIC_GISCUS_LANG` (default: `en`)

## GitHub Pages deployment

1. Push to the `main` branch.
2. In GitHub repository settings, set **Pages** source to **GitHub Actions**.
3. The workflow in `.github/workflows/deploy.yml` builds and deploys `dist/`.
