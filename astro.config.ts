import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import tailwindcss from '@tailwindcss/vite';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeSlug from 'rehype-slug';

export default defineConfig({
  site: 'https://snakamoto404.github.io',
  output: 'static',
  integrations: [
    sitemap({
      filter: (page) => {
        const pathname = new URL(page).pathname;
        return !['/subscribe/', '/subscribed/', '/unsubscribe/', '/unsubscribed/'].includes(pathname);
      }
    })
  ],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeSlug, rehypeKatex]
  },
  vite: {
    plugins: [tailwindcss()]
  }
});
