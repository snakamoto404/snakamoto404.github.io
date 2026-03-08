const PROD_HIDDEN_BLOG_SLUGS = new Set([
  'machine-learning/ot-generative-0-static',
  'machine-learning/ot-generative-1-wasserstein-geometry',
  'machine-learning/ot-generative-2-drifting-models',
  'machine-learning/ot-generative-3-maximum-likelihood-smooth-drifting',
  'machine-learning/ot-generative-4-one-step-generation'
]);

export const isBlogHiddenInProduction = (slug: string): boolean =>
  import.meta.env.PROD && PROD_HIDDEN_BLOG_SLUGS.has(slug);

export const filterVisibleBlogs = <T extends { slug: string }>(entries: T[]): T[] =>
  entries.filter((entry) => !isBlogHiddenInProduction(entry.slug));
