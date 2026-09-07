const PROD_HIDDEN_BLOG_SLUGS = new Set([
  'machine-learning/ot-generative-4-one-step-generation'
]);

// Deprecated posts stay in the repository but have no page or listing.
const DEPRECATED_BLOG_SLUGS = new Set([
  'film-reviews/2026-02-15-hamnet'
]);

export const isBlogHiddenInProduction = (slug: string): boolean =>
  import.meta.env.PROD && PROD_HIDDEN_BLOG_SLUGS.has(slug);

export const filterVisibleBlogs = <T extends { slug: string }>(entries: T[]): T[] =>
  entries.filter((entry) =>
    !DEPRECATED_BLOG_SLUGS.has(entry.slug) && !isBlogHiddenInProduction(entry.slug)
  );
