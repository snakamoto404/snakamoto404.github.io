# snakamoto404.github.io

Astro static site scaffolded for GitHub Pages user-site hosting.

## Features

- Home page at `/`
- Archive index at `/archive/`
- Daily archive pages at `/archive/YYYY-MM-DD/` from markdown content collection
- Subscribe and unsubscribe form pages at `/subscribe/` and `/unsubscribe/`
- Giscus comments component on each daily archive page
- GitHub Actions workflow for Pages deployment

## Local setup

1. Install dependencies:

```bash
npm install
```

2. Run dev server:

```bash
npm run dev
```

3. Build for production:

```bash
npm run build
```

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
